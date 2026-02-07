import pandas as pd
import numpy as np
import nfl_data_py as nfl

DEFAULT_ROLLING_SPAN = 10  # EWM span (replaces simple rolling window)
MIN_PERIODS = 3            # minimum games before producing a rolling value


def add_targets(df):
    """Calculates target variables (Cover, Margin) from raw scores.
    Pushes (exact ties vs spread) are coded as NaN so they are excluded."""
    df = df.sort_values(["season", "week", "date"]).copy()
    df["home_margin"] = df["home_score"] - df["away_score"]
    df["home_cover"] = np.where(
        df["home_margin"] == df["spread_home"],
        np.nan,  # push = exclude
        (df["home_margin"] > df["spread_home"]).astype(float),
    )
    return df


def build_rolling_team_features(df, span=DEFAULT_ROLLING_SPAN):
    """
    Fetches Play-by-Play data to build advanced EPA metrics.
    Uses EWM (exponentially weighted moving average) instead of simple rolling.
    Splits EPA into pass and rush components. Adds CPOE.
    """
    seasons = df["season"].unique().tolist()
    print(f"Fetching Play-by-Play data for seasons: {seasons} (This may take a moment...)")

    cols = [
        "game_id", "season", "week", "home_team", "away_team",
        "posteam", "defteam", "epa", "success", "play_type", "cpoe",
    ]

    try:
        pbp = nfl.import_pbp_data(seasons, columns=cols)
    except Exception as e:
        print(f"Error fetching PBP data: {e}")
        return df

    pbp = pbp.dropna(subset=["epa", "posteam", "defteam"])
    pbp = pbp[pbp["play_type"].isin(["pass", "run"])]

    # --- Game-level stats split by pass / rush ---
    # Offense
    off_stats = (
        pbp.groupby(["season", "week", "posteam", "play_type"])
        .agg(epa=("epa", "mean"), success=("success", "mean"))
        .reset_index()
    )
    off_pivot = off_stats.pivot_table(
        index=["season", "week", "posteam"],
        columns="play_type",
        values=["epa", "success"],
    )
    off_pivot.columns = [f"off_{val}_{pt}" for val, pt in off_pivot.columns]
    off_pivot = off_pivot.reset_index().rename(columns={"posteam": "team"})

    # Overall offensive EPA & success rate (needed for sr_diff)
    off_overall = (
        pbp.groupby(["season", "week", "posteam"])
        .agg(off_epa=("epa", "mean"), off_success=("success", "mean"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    # CPOE (only on pass plays with non-null cpoe)
    cpoe_stats = (
        pbp[pbp["play_type"] == "pass"]
        .dropna(subset=["cpoe"])
        .groupby(["season", "week", "posteam"])
        .agg(off_cpoe=("cpoe", "mean"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    # Defense
    def_stats = (
        pbp.groupby(["season", "week", "defteam", "play_type"])
        .agg(epa=("epa", "mean"), success=("success", "mean"))
        .reset_index()
    )
    def_pivot = def_stats.pivot_table(
        index=["season", "week", "defteam"],
        columns="play_type",
        values=["epa", "success"],
    )
    def_pivot.columns = [f"def_{val}_{pt}" for val, pt in def_pivot.columns]
    def_pivot = def_pivot.reset_index().rename(columns={"defteam": "team"})

    # Overall defensive EPA & success rate
    def_overall = (
        pbp.groupby(["season", "week", "defteam"])
        .agg(def_epa=("epa", "mean"), def_success=("success", "mean"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    # Merge all into one team-game row
    team_game = off_pivot.merge(off_overall, on=["season", "week", "team"], how="outer")
    team_game = team_game.merge(cpoe_stats, on=["season", "week", "team"], how="left")
    team_game = team_game.merge(def_pivot, on=["season", "week", "team"], how="outer")
    team_game = team_game.merge(def_overall, on=["season", "week", "team"], how="outer")

    # --- EWM rolling averages (shift first to avoid leakage) ---
    team_game = team_game.sort_values(["team", "season", "week"])

    metrics = [c for c in team_game.columns if c.startswith(("off_", "def_"))]
    for m in metrics:
        team_game[f"roll_{m}"] = team_game.groupby("team")[m].transform(
            lambda x: x.shift(1).ewm(span=span, min_periods=MIN_PERIODS).mean()
        )

    # --- Merge back to main schedule: home side ---
    roll_cols = [c for c in team_game.columns if c.startswith("roll_")]
    merge_cols = ["season", "week", "team"] + roll_cols

    df = df.merge(
        team_game[merge_cols],
        left_on=["season", "week", "home_team"],
        right_on=["season", "week", "team"],
        how="left",
    )
    rename_home = {c: c.replace("roll_", "home_") for c in roll_cols}
    df = df.rename(columns=rename_home).drop(columns=["team"])

    # --- Merge: away side ---
    df = df.merge(
        team_game[merge_cols],
        left_on=["season", "week", "away_team"],
        right_on=["season", "week", "team"],
        how="left",
        suffixes=("", "_dup"),
    )
    rename_away = {c: c.replace("roll_", "away_") for c in roll_cols}
    df = df.rename(columns=rename_away).drop(columns=["team"], errors="ignore")
    # drop any _dup columns from suffix collision
    df = df.drop(columns=[c for c in df.columns if c.endswith("_dup")], errors="ignore")

    # DO NOT fill NaN with 0 — let dropna() in train.py handle it
    return df


def add_rest_and_division(df):
    """Add rest-day differential and divisional-game flag.
    Uses nflverse's built-in home_rest/away_rest/div_game columns when available,
    falls back to computing them manually otherwise."""
    df = df.copy()

    # --- Rest days ---
    if "home_rest" in df.columns and "away_rest" in df.columns:
        # nflverse already provides these
        df["rest_diff"] = df["home_rest"] - df["away_rest"]
    else:
        # Compute from game dates
        df["date"] = pd.to_datetime(df["date"])
        home = df[["season", "week", "home_team", "date"]].rename(columns={"home_team": "team"})
        away = df[["season", "week", "away_team", "date"]].rename(columns={"away_team": "team"})
        team_dates = pd.concat([home, away], ignore_index=True).sort_values(["team", "season", "date"])
        team_dates = team_dates.drop_duplicates(subset=["team", "season", "week"])
        team_dates["rest_days"] = team_dates.groupby("team")["date"].diff().dt.days.clip(upper=14)

        rest_lookup = team_dates[["season", "week", "team", "rest_days"]]
        df = df.merge(
            rest_lookup.rename(columns={"team": "home_team", "rest_days": "home_rest"}),
            on=["season", "week", "home_team"], how="left",
        )
        df = df.merge(
            rest_lookup.rename(columns={"team": "away_team", "rest_days": "away_rest"}),
            on=["season", "week", "away_team"], how="left",
        )
        df["rest_diff"] = df["home_rest"] - df["away_rest"]

    # --- Divisional game flag ---
    if "div_game" not in df.columns:
        _div_map = _build_division_map()
        df["home_div"] = df["home_team"].map(_div_map)
        df["away_div"] = df["away_team"].map(_div_map)
        df["div_game"] = (df["home_div"] == df["away_div"]).astype(int)
        df = df.drop(columns=["home_div", "away_div"], errors="ignore")

    return df


def _build_division_map():
    """Static mapping of NFL teams to divisions."""
    divisions = {
        "AFC East": ["BUF", "MIA", "NE", "NYJ"],
        "AFC North": ["BAL", "CIN", "CLE", "PIT"],
        "AFC South": ["HOU", "IND", "JAX", "TEN"],
        "AFC West": ["DEN", "KC", "LV", "LAC", "OAK", "SD"],
        "NFC East": ["DAL", "NYG", "PHI", "WAS"],
        "NFC North": ["CHI", "DET", "GB", "MIN"],
        "NFC South": ["ATL", "CAR", "NO", "TB"],
        "NFC West": ["ARI", "LAR", "SEA", "SF", "LA", "STL"],
    }
    team_to_div = {}
    for div, teams in divisions.items():
        for t in teams:
            team_to_div[t] = div
    return team_to_div


def compute_epa_features(df):
    """Derives model features from rolling EPA columns. Call after build_rolling_team_features()."""
    df = df.copy()

    # Helper to safely get column or zeros
    def _col(name):
        if name in df.columns:
            return df[name]
        return pd.Series(0.0, index=df.index)

    # --- Pass EPA differential ---
    home_pass_net = _col("home_off_epa_pass") - _col("home_def_epa_pass")
    away_pass_net = _col("away_off_epa_pass") - _col("away_def_epa_pass")
    df["pass_epa_diff"] = home_pass_net - away_pass_net

    # --- Rush EPA differential ---
    home_rush_net = _col("home_off_epa_run") - _col("home_def_epa_run")
    away_rush_net = _col("away_off_epa_run") - _col("away_def_epa_run")
    df["rush_epa_diff"] = home_rush_net - away_rush_net

    # --- Pass matchup advantage ---
    home_pass_matchup = _col("home_off_epa_pass") - _col("away_def_epa_pass")
    away_pass_matchup = _col("away_off_epa_pass") - _col("home_def_epa_pass")
    df["matchup_pass"] = home_pass_matchup - away_pass_matchup

    # --- Success rate differential ---
    df["sr_diff"] = (
        (_col("home_off_success") - _col("home_def_success"))
        - (_col("away_off_success") - _col("away_def_success"))
    )

    # --- CPOE differential ---
    df["cpoe_diff"] = _col("home_off_cpoe") - _col("away_off_cpoe")

    # --- Market line features ---
    df["spread_home_feat"] = df["spread_home"]
    df["spread_sq"] = df["spread_home"] ** 2

    # --- Key number indicator (spread near 3 or 7) ---
    abs_spread = df["spread_home"].abs()
    df["key_number"] = (
        ((abs_spread >= 2.5) & (abs_spread <= 3.5))
        | ((abs_spread >= 6.5) & (abs_spread <= 7.5))
    ).astype(int)

    # --- Rest diff and div_game should already be on df from add_rest_and_division ---
    # Fill rest_diff NaN (week 1) with 0 — all teams have equal rest
    if "rest_diff" in df.columns:
        df["rest_diff"] = df["rest_diff"].fillna(0)
    else:
        df["rest_diff"] = 0

    if "div_game" not in df.columns:
        df["div_game"] = 0

    return df
