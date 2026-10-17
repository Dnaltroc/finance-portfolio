"""Live tracker math: how far the price has moved, and how far it is from the memo's numbers."""

from __future__ import annotations


def pct_change(new: float | None, ref: float | None) -> float | None:
    """new / ref - 1, or None when either side is missing or the reference is zero (NM)."""
    if new is None or not ref:
        return None
    return new / ref - 1


def tracker_moves(price: float, previous_close: float | None, memo_price: float, target: float) -> dict:
    """Day move, move since the memo was priced, and the move still needed to reach the target."""
    return {
        "day_change": None if previous_close is None else price - previous_close,
        "day_pct": pct_change(price, previous_close),
        "since_memo_pct": pct_change(price, memo_price),
        "to_target_pct": pct_change(target, price),
    }
