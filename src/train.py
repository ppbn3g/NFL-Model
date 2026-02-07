import os
import warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from features import (
    add_targets, build_rolling_team_features, compute_epa_features,
    add_rest_and_division, DEFAULT_ROLLING_SPAN,
)
from backtest import backtest_spread

# Try importing LightGBM; fall back to logistic-only if unavailable
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    warnings.warn("LightGBM not installed — using logistic regression only.")

# Feature columns used by the model (orthogonal pass/rush split, no multicollinearity)
SPREAD_FEATURES = [
    "pass_epa_diff",      # (Home Pass Net EPA) - (Away Pass Net EPA)
    "rush_epa_diff",      # (Home Rush Net EPA) - (Away Rush Net EPA)
    "matchup_pass",       # Home pass off vs Away pass def matchup
    "sr_diff",            # Success Rate Differential
    "cpoe_diff",          # CPOE differential
    "spread_home_feat",   # Market Line
    "spread_sq",          # Spread squared (nonlinearity at extremes)
    "key_number",         # 1 if spread near 3 or 7
    "rest_diff",          # Home rest days - Away rest days
    "div_game",           # Divisional game flag
]


def load_nflfastR_games():
    url = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
    df = pd.read_csv(url)
    df = df.loc[:, ~df.columns.duplicated()]
    if "gameday" in df.columns:
        df = df.rename(columns={"gameday": "date"})
    if "spread_line" in df.columns:
        df = df.rename(columns={"spread_line": "spread_home"})
    return df


def _prepare_data():
    """Load games, build features, and return cleaned dataframe."""
    print("--- Loading & Building Features ---")
    df = load_nflfastR_games()
    df = add_targets(df)
    df = add_rest_and_division(df)
    df = build_rolling_team_features(df, span=DEFAULT_ROLLING_SPAN)
    df = compute_epa_features(df)
    # Drop rows where features or target are missing (early-season NaN)
    df_clean = df.dropna(subset=SPREAD_FEATURES + ["home_cover"])
    return df, df_clean


def _tune_C(X_train, y_train, n_splits=3):
    """Tune logistic regression C using TimeSeriesSplit within the training fold."""
    C_grid = [0.01, 0.1, 0.5, 1.0, 5.0]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_C = 0.5
    best_loss = float("inf")

    for C in C_grid:
        fold_losses = []
        for train_idx, val_idx in tscv.split(X_train):
            Xtr, Xval = X_train[train_idx], X_train[val_idx]
            ytr, yval = y_train[train_idx], y_train[val_idx]

            lr = LogisticRegression(C=C, solver="liblinear", max_iter=1000)
            lr.fit(Xtr, ytr)
            probs = lr.predict_proba(Xval)[:, 1]
            fold_losses.append(log_loss(yval, probs))

        mean_loss = np.mean(fold_losses)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_C = C

    return best_C


def _build_ensemble(X_train, y_train, X_test, best_C):
    """Build logistic regression + optional LightGBM ensemble with calibration."""
    # --- Logistic Regression ---
    lr = LogisticRegression(C=best_C, solver="liblinear", max_iter=1000)
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_test)[:, 1]

    if not HAS_LGBM:
        return lr_probs, lr, None

    # --- LightGBM (conservative) ---
    lgb_model = lgb.LGBMClassifier(
        max_depth=3,
        num_leaves=8,
        n_estimators=100,
        min_child_samples=50,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(X_train, y_train)
    lgb_probs = lgb_model.predict_proba(X_test)[:, 1]

    # 50/50 ensemble
    ensemble_probs = 0.5 * lr_probs + 0.5 * lgb_probs
    return ensemble_probs, lr, lgb_model


def train_model():
    print("--- Training Advanced EPA Model ---")
    _, df_clean = _prepare_data()

    # Train/Test Split
    test_start = df_clean["season"].max() - 1
    train = df_clean[df_clean["season"] < test_start].copy()
    test = df_clean[df_clean["season"] >= test_start].copy()

    print(f"Train Size: {len(train)}, Test Size: {len(test)}")

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[SPREAD_FEATURES])
    X_test = scaler.transform(test[SPREAD_FEATURES])
    y_train = train["home_cover"].values

    # Tune C
    best_C = _tune_C(X_train, y_train)
    print(f"Best C: {best_C}")

    # Build model(s)
    probs, lr_model, lgb_model = _build_ensemble(X_train, y_train, X_test, best_C)
    test["p_home"] = probs

    auc = roc_auc_score(test["home_cover"], probs)
    print(f"Test AUC: {auc:.4f}")
    print(f"LR Coefficients: {dict(zip(SPREAD_FEATURES, lr_model.coef_[0]))}")

    # Save
    os.makedirs("models", exist_ok=True)
    joblib.dump({"lr": lr_model, "lgb": lgb_model, "scaler": scaler,
                 "features": SPREAD_FEATURES}, "models/model_spread.pkl")

    # Backtest
    bets, summary = backtest_spread(test, min_edge=0.01)
    print(f"\n--- Backtest Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if not bets.empty:
        os.makedirs("data", exist_ok=True)
        bets.to_csv("data/backtest_bets.csv", index=False)
        print(f"  Saved {len(bets)} bets to data/backtest_bets.csv")


def walk_forward_backtest(min_edge=0.01):
    """
    Walk-forward expanding-window backtest.
    For each season S (starting after 3 training seasons):
      - Train on all seasons before S
      - Predict season S
      - Collect bets
    Features are scaled per-fold. C is tuned per-fold.
    """
    print("--- Walk-Forward Backtest ---")
    _, df_clean = _prepare_data()

    seasons = sorted(df_clean["season"].unique())
    min_train_seasons = 3

    all_bets = []
    all_summaries = []
    all_predicted = []  # collect full predicted dataframes for sensitivity

    for test_season in seasons[min_train_seasons:]:
        train_data = df_clean[df_clean["season"] < test_season]
        test_data = df_clean[df_clean["season"] == test_season].copy()

        if len(train_data) < 100 or len(test_data) < 10:
            continue

        # Scale within fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_data[SPREAD_FEATURES])
        X_test = scaler.transform(test_data[SPREAD_FEATURES])
        y_train = train_data["home_cover"].values

        # Tune C within fold
        best_C = _tune_C(X_train, y_train)

        # Build ensemble
        probs, _, _ = _build_ensemble(X_train, y_train, X_test, best_C)
        test_data["p_home"] = probs

        all_predicted.append(test_data)

        bets, summary = backtest_spread(test_data, min_edge=min_edge)

        if not bets.empty:
            all_bets.append(bets)
            summary["season"] = test_season
            all_summaries.append(summary)
            print(f"  Season {test_season}: {summary['bets']} bets, "
                  f"hit rate {summary['hit_rate']:.1%}, PnL {summary['total_pnl']:+.2f}, "
                  f"C={best_C}")

    if all_bets:
        all_bets_df = pd.concat(all_bets, ignore_index=True)
        all_predicted_df = pd.concat(all_predicted, ignore_index=True)
        return all_bets_df, pd.DataFrame(all_summaries), all_predicted_df
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


if __name__ == "__main__":
    train_model()
