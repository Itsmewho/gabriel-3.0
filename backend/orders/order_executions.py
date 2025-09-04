from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BEParams:
    pip_size: float
    trigger_pips: float
    offset_pips: float


@dataclass(frozen=True)
class TSLParams:
    pip_size: float
    distance_points: float  # points in "pips" terms from your config


# --- Core helpers ---
def compute_break_even_sl(
    *, direction: str, price_open: float, high: float, low: float, be: BEParams
) -> Optional[float]:
    """Return desired BE SL or None if not triggered.
    direction: 'buy' | 'sell'
    """
    move_pips = (
        (high - price_open) if direction == "buy" else (price_open - low)
    ) / be.pip_size
    if move_pips < be.trigger_pips:
        return None
    offset = be.offset_pips * be.pip_size
    return price_open + (offset if direction == "buy" else -offset)


def compute_trailing_sl(
    *,
    direction: str,
    price_open: float,
    high: float,
    low: float,
    current_sl: float,
    tsl: TSLParams
) -> Optional[float]:
    """Tighten SL only if in profit beyond distance; return tighter level or None.
    distance_points are in 'points' the same unit you use elsewhere (pips here).
    """
    distance = tsl.distance_points * tsl.pip_size
    if direction == "buy":
        if (high - price_open) <= distance:
            return None
        desired = max(current_sl or 0.0, high - distance)
        return desired if desired > (current_sl or 0.0) else None
    else:
        if (price_open - low) <= distance:
            return None
        desired = min(current_sl or 1e12, low + distance)
        return desired if desired < (current_sl or 1e12) else None


def respects_stops(*, ref_price: float, level: float, stops_level: float) -> bool:
    if not stops_level:
        return True
    return abs(ref_price - level) >= stops_level


def clamp_to_stops(
    *, direction: str, desired_level: float, bid: float, ask: float, stops_level: float
) -> float:
    """If desired level violates broker 'stops level', clamp to nearest valid.
    For buys, SL must be <= bid - stops_level. For sells, SL must be >= ask + stops_level.
    """
    if stops_level <= 0:
        return desired_level
    eps = 1e-9
    if direction == "buy":
        return min(desired_level, bid - stops_level - eps)
    else:
        return max(desired_level, ask + stops_level + eps)
