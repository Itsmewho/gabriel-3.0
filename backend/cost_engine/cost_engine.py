from typing import Dict, Any


class CostModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pip_size = 0.0001 if "JPY" not in config.get("SYMBOL", "") else 0.01

    def get_spread_cost(self, lot_size: float) -> float:
        if not self.config.get("USE_SPREAD_COST", False):
            return 0.0
        spread_pips = self.config.get("SPREAD_PIPS", 0.2)
        value_per_pip = self.config.get("VALUE_PER_PIP_PER_LOT", 10.0)
        return spread_pips * value_per_pip * lot_size

    def get_commission(self, lot_size: float) -> float:
        commission_per_lot = self.config.get("COMMISSION_PER_LOT_RT", 7.0)
        return commission_per_lot * lot_size

    def get_swap_cost(self, direction: str, lot_size: float, weekday: int) -> float:
        if not self.config.get("USE_SWAP_COST", False):
            return 0.0
        swap_long = self.config.get("SWAP_LONG_POINTS", -9.89)
        swap_short = self.config.get("SWAP_SHORT_POINTS", 5.44)
        daily_rate = swap_long if direction == "buy" else swap_short
        multiplier = 3 if weekday == 2 else 1  # Wed triple
        return (daily_rate * multiplier) * lot_size
