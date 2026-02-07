# src/odds.py

def american_to_implied_prob(odds: int) -> float:
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)

def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds."""
    if odds < 0:
        return 1 + (100 / abs(odds))
    return 1 + (odds / 100)

def profit_per_unit(odds: int) -> float:
    # profit if you stake 1 unit and win
    if odds < 0:
        return 100.0 / (-odds)
    return odds / 100.0
