import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path
from utils.helpers import setup_logger

logger = setup_logger(__name__)


# --- Helper stats ---
def _safe_sharpe(daily_ret: pd.Series) -> float:
    if daily_ret.empty:
        return float("nan")
    vol = daily_ret.std()
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return float((daily_ret.mean() / vol) * np.sqrt(252))


def _safe_sortino(daily_ret: pd.Series) -> float:
    if daily_ret.empty:
        return float("nan")
    downside = daily_ret[daily_ret < 0]
    dvol = downside.std()
    if dvol == 0 or np.isnan(dvol):
        return float("nan")
    return float((daily_ret.mean() / dvol) * np.sqrt(252))


def _cagr(equity_ts: pd.Series) -> float:
    if equity_ts.empty or len(equity_ts) < 2:
        return 0.0
    start_val = float(equity_ts.iloc[0])
    end_val = float(equity_ts.iloc[-1])
    if start_val <= 0:
        return 0.0
    delta_days = (equity_ts.index[-1] - equity_ts.index[0]).days  # type: ignore
    years = max(delta_days / 365.25, 1 / 365.25)
    return float((end_val / start_val) ** (1 / years) - 1)


# --- Public API ---
def get_performance_report(
    self,
) -> Optional[Dict[str, Any]]:  # 'self' refers to the OrderEngine instance
    if not getattr(self, "trade_history", None):
        return None

    history_df = pd.DataFrame(self.trade_history)
    if history_df.empty:
        return None

    history_df["entry_time"] = pd.to_datetime(history_df["entry_time"])
    history_df["exit_time"] = pd.to_datetime(history_df["exit_time"])
    history_df["duration"] = history_df["exit_time"] - history_df["entry_time"]

    total_trades = len(history_df)
    wins = history_df[history_df["pnl"] > 0]
    losses = history_df[history_df["pnl"] <= 0]
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0.0
    total_pnl = float(history_df["pnl"].sum())
    gross_profit = float(wins["pnl"].sum()) if not wins.empty else 0.0
    gross_loss = float(abs(losses["pnl"].sum())) if not losses.empty else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    equity_df = pd.DataFrame(self.equity_curve)
    max_drawdown, max_drawdown_pct, sharpe, sortino, cagr, calmar = (
        0.0,
        0.0,
        float("nan"),
        float("nan"),
        0.0,
        float("nan"),
    )

    if not equity_df.empty:
        equity_df["time"] = pd.to_datetime(equity_df["time"])
        equity_df.sort_values("time", inplace=True)
        equity_df["dd"] = equity_df["equity"].cummax() - equity_df["equity"]
        max_drawdown = float(equity_df["dd"].max())
        max_equity = float(equity_df["equity"].cummax().max())
        if max_equity > 0:
            max_drawdown_pct = (max_drawdown / max_equity) * 100
        eq_ts = equity_df.set_index("time")["equity"]
        daily_eq = eq_ts.resample("1D").last().ffill()
        daily_ret = daily_eq.pct_change().dropna()
        sharpe = _safe_sharpe(daily_ret)
        sortino = _safe_sortino(daily_ret)
        cagr = _cagr(daily_eq)
        calmar = (
            (cagr / (max_drawdown_pct / 100)) if max_drawdown_pct > 0 else float("inf")
        )

    # --- MODIFIED: Refined Stop Loss Counters ---
    sl_exits = history_df[history_df["exit_reason"] == "Stop Loss"]
    losing_sl_hits = len(sl_exits[sl_exits["pnl"] < 0])
    profitable_ts_hits = len(sl_exits[sl_exits["pnl"] >= 0])
    # --- END MODIFICATION ---

    tp_hits = len(history_df[history_df["exit_reason"] == "Take Profit"])
    be_activations = (
        int(history_df["be_activated"].sum())
        if "be_activated" in history_df.columns
        else 0
    )
    ts_activations = (
        int(history_df["ts_activated"].sum())
        if "ts_activated" in history_df.columns
        else 0
    )

    report: Dict[str, Any] = {
        "strategy_name": self.strategy.get_name(),
        "initial_balance": float(self.config["INITIAL_BALANCE"]),
        "final_balance": float(self.balance),
        "net_profit": total_pnl,
        "total_trades": int(total_trades),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "max_drawdown_abs": float(max_drawdown),
        "max_drawdown_pct": float(max_drawdown_pct),
        "avg_win": float(wins["pnl"].mean()) if not wins.empty else 0.0,
        "avg_loss": float(losses["pnl"].mean()) if not losses.empty else 0.0,
        "margin_calls": (
            int(len(history_df[history_df["exit_reason"] == "Margin Call"]))
            if "exit_reason" in history_df.columns
            else 0
        ),
        "max_concurrent_trades": max(getattr(self, "concurrent_trades_log", []) or [0]),
        # --- MODIFIED: Updated stats in the report dictionary ---
        "losing_sl_hits": losing_sl_hits,
        "profitable_ts_hits": profitable_ts_hits,
        "tp_hits": tp_hits,
        "be_activations": be_activations,
        "ts_activations": ts_activations,
        "blowout_activations": getattr(self, "blowout_activations", 0),
        "insufficient_funds_attempts": getattr(self, "insufficient_funds_attempts", 0),
        # --- END MODIFICATION ---
        "max_consecutive_losses": int(getattr(self, "max_consecutive_losses", 0)),
        "max_consecutive_loss_amount": float(
            getattr(self, "max_consecutive_loss_amount", 0.0)
        ),
        "max_consecutive_wins": int(getattr(self, "max_consecutive_wins", 0)),
        "max_consecutive_win_amount": float(
            getattr(self, "max_consecutive_win_amount", 0.0)
        ),
        "daily_sharpe": sharpe,
        "daily_sortino": sortino,
        "cagr": cagr,
        "calmar": calmar,
    }
    return report


def print_performance_report(self, report: Dict[str, Any]):
    print("\n" + "=" * 50)
    print(f"--- Backtest Performance Report for: {report['strategy_name']} ---")
    print("=" * 50)
    print(
        f"Initial Balance: ${report['initial_balance']:.2f}, Final Balance: ${report['final_balance']:.2f}"
    )
    print(f"Net Profit/Loss: ${report['net_profit']:.2f}")
    print(
        f"Total Trades: {report['total_trades']}, Win Rate: {report['win_rate']:.2f}%, Profit Factor: {report['profit_factor']:.2f}"
    )
    print(
        f"Maximum Drawdown: ${report['max_drawdown_abs']:.2f} ({report['max_drawdown_pct']:.2f}%)"
    )
    print(
        f"Average Win: ${report['avg_win']:.2f}, Average Loss: ${report['avg_loss']:.2f}"
    )
    print("-" * 35)
    print("--- Risk & Return ---")
    print(f"Daily Sharpe: {report['daily_sharpe']:.3f}")
    print(f"Daily Sortino: {report['daily_sortino']:.3f}")
    print(f"CAGR: {report['cagr']:.2%}")
    print(f"Calmar: {report['calmar']:.3f}")
    print("-" * 35)
    print("--- Risk & Sizing Metrics ---")
    print(
        f"Max Consecutive Losses: {report['max_consecutive_losses']} (Total Loss: ${abs(report['max_consecutive_loss_amount']):.2f})"
    )
    print(
        f"Max Consecutive Wins: {report['max_consecutive_wins']} (Total Profit: ${report['max_consecutive_win_amount']:.2f})"
    )
    print(f"Max Concurrent Trades: {report['max_concurrent_trades']}")
    print("-" * 35)
    print("--- Trade Exit & Risk Events ---")
    print(f"Losing SL Hits: {report.get('losing_sl_hits', 0)}")
    print(f"Profitable Trailing SL Hits: {report.get('profitable_ts_hits', 0)}")
    print(f"Take Profit Hits: {report.get('tp_hits', 0)}")
    print(f"Break-Even Activations: {report.get('be_activations', 0)}")
    print(f"Trailing Stop Activations: {report.get('ts_activations', 0)}")
    print(f"Margin Calls: {report['margin_calls']}")
    print(f"Blowout Activations: {report.get('blowout_activations', 0)}")
    print(
        f"Insufficient Funds Attempts: {report.get('insufficient_funds_attempts', 0)}"
    )
    print("=" * 50 + "\n")


def plot_equity_curve(self, report: Dict[str, Any]):
    strategy_name = report["strategy_name"]
    base_filename = (
        f"{strategy_name.replace(' ', '_').replace('/', '_').lower()}_equity"
    )
    results_path = Path("results")
    results_path.mkdir(exist_ok=True)

    equity_df = pd.DataFrame(self.equity_curve)
    if equity_df.empty or len(equity_df) < 2:
        logger.warning(
            f"Cannot plot equity curve for {strategy_name}: Not enough equity data."
        )
        return

    equity_df["time"] = pd.to_datetime(equity_df["time"])
    equity_df.set_index("time", inplace=True)

    def _save_plot(df_slice: pd.DataFrame, title_suffix: str, file_suffix: str):
        """Helper function to generate and save a plot for a specific period."""
        if df_slice.empty:
            logger.warning(
                f"Skipping '{title_suffix}' plot for {strategy_name}: No data in this period."
            )
            return

        plt.figure(figsize=(12, 6))
        plt.plot(df_slice.index, df_slice["equity"], label="Equity")
        plt.title(f"Equity Curve for: {strategy_name} ({title_suffix})")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        plt.tight_layout()

        filepath = results_path / f"{base_filename}_{file_suffix}.png"
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Equity curve plot saved to {filepath}")

    # 1. Plot the full curve
    _save_plot(equity_df, "Full Period", "full")

    # 2. Plot the first month
    start_date = equity_df.index.min()
    first_month_end = start_date + pd.DateOffset(months=1)
    first_month_df = equity_df[equity_df.index <= first_month_end]
    _save_plot(first_month_df, "First Month", "first_month")

    # 3. Plot a month in the middle
    end_date = equity_df.index.max()
    total_duration_days = (end_date - start_date).days
    if total_duration_days > 60:  # Ensure there's a distinct middle period
        mid_point_date = start_date + pd.Timedelta(days=total_duration_days // 2)
        middle_month_start = mid_point_date - pd.Timedelta(days=15)
        middle_month_end = mid_point_date + pd.Timedelta(days=15)
        middle_month_df = equity_df[
            (equity_df.index >= middle_month_start)
            & (equity_df.index <= middle_month_end)
        ]
        _save_plot(middle_month_df, "Middle Month", "middle_month")

    # 4. Plot the last month
    if total_duration_days > 30:  # Ensure there's at least a month of data
        last_month_start = end_date - pd.DateOffset(months=1)
        last_month_df = equity_df[equity_df.index >= last_month_start]
        _save_plot(last_month_df, "Last Month", "last_month")
