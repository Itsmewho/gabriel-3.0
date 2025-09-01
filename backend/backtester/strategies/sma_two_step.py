from __future__ import annotations
from typing import Any, Optional, Sequence, Dict
import math
import pandas as pd

from backtester.strategies.base_strat import BaseStrategy
from backtester.account_management.types import SizeRequest


def _round_to_step(x: float, step: float, min_lot: float, max_lot: float) -> float:
    lots = max(0.0, math.floor(x / step) * step)
    return max(min_lot, min(lots, max_lot))


class EmaTwoStepSignal(BaseStrategy):
    """
    Two-step crossover strategy using a last-cross ledger and a final state check.

    This version confirms a trade by ensuring two conditions are met:
    1. Event Check: All required crossover events have occurred within the timeframe.
    2. State Check: At the moment of execution, all fast lines are currently on the
       correct side of their respective signal lines.

    SELL:
      - Stage 1: For each fast line, it must cross below BOTH EMA150 and EMA200.
      - Stage 2: For each fast line, it must ALSO cross below EMA450.
      - Condition: The event sequence must complete within CONFIRM_WINDOW_BARS, and
                   at the time of the signal, all fast lines must be below all signal lines.
    """

    def __init__(
        self,
        symbol: str,
        config: dict[str, Any],
        strat_cfg=None,
        governor=None,
        magic: Optional[int] = 7701,
    ):
        super().__init__(symbol, config, strat_cfg, governor=governor)

        # --- Lines ---
        self.trio: Sequence[str] = config.get(
            "FAST_PACK", ["sma_high_30", "sma_low_30", "ema_50"]
        )
        self.ema150 = config.get("EMA150_COL", "ema_150")
        self.ema200 = config.get("EMA200_COL", "ema_200")
        self.ema450 = config.get("EMA450_COL", "ema_450")
        self.all_signal_lines = [self.ema150, self.ema200, self.ema450]

        # --- Stage Definitions ---
        self.stage1_lines: Dict[str, Sequence[str]] = {
            "sell": [self.ema150, self.ema200],
            "buy": [self.ema450],
        }
        self.stage2_lines: Dict[str, Sequence[str]] = {
            "sell": [self.ema450],
            "buy": [self.ema150, self.ema200],
        }

        # --- Timing & Risk ---
        self.TIMEFRAME_BARS = int(config.get("CONFIRM_WINDOW_BARS", 30))
        self.cooldown_bars = int(config.get("COOLDOWN_BARS", 5))
        self.sl_pips = float(config.get("SL_PIPS", 10))
        self.tp_pips = float(config.get("TP_PIPS", 50))
        self.eps = float(config.get("EPS", 0.0) or 0.0)

        # --- Runtime State & Ledger ---
        self.prev_row: Optional[pd.Series] = None
        self.cooldown = 0
        self.parent_df: Optional[pd.DataFrame] = None
        self.magic = magic
        self.bar_pos = -1

        self.last_cross_pos: Dict[str, Dict[str, int]] = {f: {} for f in self.trio}
        self.last_cross_dir: Dict[str, Dict[str, str]] = {f: {} for f in self.trio}

    # ---------- Utilities ----------
    def bind_parent_df(self, df: pd.DataFrame):
        self.parent_df = df
        return self

    def _open_count_for_me(self, broker) -> int:
        return sum(
            getattr(tr, "strategy_id", None) == self.name for tr in broker.open_trades
        )

    def _have_cols(self, row: pd.Series) -> bool:
        need = set(self.trio) | set(self.all_signal_lines) | {"close"}
        return all(c in row.index for c in need)

    def _check_cross(
        self, fast_val: float, prev_fast: float, sig_val: float, prev_sig: float
    ) -> Optional[str]:
        if prev_fast <= prev_sig - self.eps and fast_val > sig_val + self.eps:
            return "buy"
        if prev_fast >= prev_sig + self.eps and fast_val < sig_val - self.eps:
            return "sell"
        return None

    def _is_on_correct_side(self, row: pd.Series, side: str) -> bool:
        """Final state check: confirms all fast lines are correctly positioned."""
        all_lines_for_side = self.stage1_lines[side] + self.stage2_lines[side]
        for fast_line in self.trio:
            for sig_line in all_lines_for_side:
                fast_val = row[fast_line]
                sig_val = row[sig_line]
                if side == "buy" and fast_val <= sig_val:
                    return False
                if side == "sell" and fast_val >= sig_val:
                    return False
        return True

    # ---------- Main Hook ----------
    def on_bar(self, broker, t, row: pd.Series):
        self.bar_pos += 1
        if self.prev_row is None or self.parent_df is None or not self._have_cols(row):
            self.prev_row = row
            return None

        if self.cooldown > 0:
            self.cooldown -= 1

        if (
            self.cooldown > 0
            or self._open_count_for_me(broker) >= self.strat_cfg.max_concurrent_trades
        ):
            self.prev_row = row
            return None

        if self.governor and not self.governor.allow_new_trade(self.name).ok:
            self.prev_row = row
            return None

        # 1. Update the ledger
        for fast_line in self.trio:
            for sig_line in self.all_signal_lines:
                cross_dir = self._check_cross(
                    row[fast_line],
                    self.prev_row[fast_line],
                    row[sig_line],
                    self.prev_row[sig_line],
                )
                if cross_dir:
                    self.last_cross_pos[fast_line][sig_line] = self.bar_pos
                    self.last_cross_dir[fast_line][sig_line] = cross_dir

        # 2. Check for trade signals
        for side in ("buy", "sell"):
            stage1_times, stage2_times = [], []
            all_conditions_met = True

            for fast_line in self.trio:
                s1_crosses = [
                    self.last_cross_pos[fast_line].get(s, -1)
                    for s in self.stage1_lines[side]
                    if self.last_cross_dir[fast_line].get(s) == side
                ]
                s2_crosses = [
                    self.last_cross_pos[fast_line].get(s, -1)
                    for s in self.stage2_lines[side]
                    if self.last_cross_dir[fast_line].get(s) == side
                ]

                if len(s1_crosses) != len(self.stage1_lines[side]) or len(
                    s2_crosses
                ) != len(self.stage2_lines[side]):
                    all_conditions_met = False
                    break

                stage1_completed_at = max(s1_crosses)
                stage2_completed_at = max(s2_crosses)

                if stage2_completed_at < stage1_completed_at:
                    all_conditions_met = False
                    break

                stage1_times.append(stage1_completed_at)
                stage2_times.append(stage2_completed_at)

            if not all_conditions_met:
                continue

            sequence_start_time = min(stage1_times)
            sequence_end_time = max(stage2_times)

            if (sequence_end_time - sequence_start_time) <= self.TIMEFRAME_BARS:
                # --- NEW: Final State Confirmation ---
                if not self._is_on_correct_side(row, side):
                    continue

                # SIGNAL CONFIRMED!
                price = float(row.get("close", row.get("Close")))
                req = SizeRequest(
                    balance=broker.balance,
                    sl_pips=self.sl_pips,
                    value_per_pip=self._value_per_pip_1lot(broker),
                )
                lots = _round_to_step(
                    self.sizer.size(req).lots,
                    broker.cfg.VOLUME_STEP,
                    broker.cfg.VOLUME_MIN,
                    9999.0,
                )

                if lots > 0:
                    tr = broker.open_trade(
                        side=side,
                        price=price,
                        wanted_lots=lots,
                        sl_pips=self.sl_pips,
                        tp_pips=self.tp_pips,
                        t=t,
                        strategy_id=self.name,
                        magic=self.magic,
                    )
                    if tr:
                        self.setup_trade(broker, tr)
                        self.cooldown = self.cooldown_bars
                        for fast_line in self.trio:
                            self.last_cross_pos[fast_line] = {}
                            self.last_cross_dir[fast_line] = {}
                        self.prev_row = row
                        return tr

        self.prev_row = row
        return None


# def run_period(
#     symbol: str, timeframe: str, start_date: str, end_date: str, seed: int | None = 42
# ) -> None:
#     if seed is not None:
#         np.random.seed(seed)

#     period_tag = f"{start_date}_{end_date}"
#     market_data = prepare_data(symbol, timeframe, start_date, end_date)
#     if market_data.empty:
#         logger.error(f"No market data for period {period_tag}. Skipping.")
#         return

#     STRATEGY_NAME = "EMA_TWO_STEP"
#     feature_spec = {
#         "ema": [50, 150, 200, 450],
#         "sma_high": [30],
#         "sma_low": [30],
#     }

#     cfg = BrokerConfig(**BACKTEST_CONFIG)
#     broker = Broker(cfg)

#     cfg_map = {
#         STRATEGY_NAME: StrategyConfig(
#             risk_mode=RiskMode.FIXED,
#             risk_pct=0.10,
#             lot_min=cfg.VOLUME_MIN,
#             lot_step=cfg.VOLUME_STEP,
#             lot_max=100.0,
#             max_risk_pct_per_trade=0.10,
#             max_drawdown_pct=0.30,
#             max_concurrent_trades=10,
#         )
#     }
#     governor = RiskGovernor(cfg_map)

#     strategies = [
#         EmaTwoStepSignal(
#             symbol=symbol,
#             config={
#                 "name": STRATEGY_NAME,
#                 "FAST_PACK": ["sma_high_30", "sma_low_30", "ema_50"],
#                 "CONFIRM_WINDOW_BARS": 180,
#                 "COOLDOWN_BARS": 0,
#                 "SL_PIPS": 10,
#                 "TP_PIPS": 50,
#                 "EPS": 0.0,
#             },
#             strat_cfg=cfg_map[STRATEGY_NAME],
#             governor=governor,
#         )
#     ]
#     if not strategies:
#         logger.error("No strategies defined for this run. Skipping.")
#         return

#     # --- Setup: Dynamic Directory Creation ---
#     strategy_folder_name = strategies[0].name
#     base_out_dir = Path(f"results/{strategy_folder_name}")
#     audit_dir = base_out_dir / "audit"
#     evals_dir = base_out_dir / "evals"
#     metrics_dir = base_out_dir / "metrics"
#     regime_dir = metrics_dir / "regime"

#     # Ensure all output directories are created
#     audit_dir.mkdir(parents=True, exist_ok=True)
#     evals_dir.mkdir(parents=True, exist_ok=True)
#     metrics_dir.mkdir(parents=True, exist_ok=True)
#     regime_dir.mkdir(parents=True, exist_ok=True)

#     # --- Backtest loop ---
#     alloc = cfg.INITIAL_BALANCE
#     allocations = {STRATEGY_NAME: alloc * 1}
#     ledger = Ledger(initial_allocations=allocations)
#     trade_to_strategy: dict[int, str] = {}
