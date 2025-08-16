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
    years = max(
        delta_days / 365.25, 1 / 365.25
    )  # avoid divide-by-zero for very short tests
    return float((end_val / start_val) ** (1 / years) - 1)


# --- Public API ---
def get_performance_report(self) -> Optional[Dict[str, Any]]:
    if not getattr(self, "trade_history", None):
        return None

    history_df = pd.DataFrame(self.trade_history)
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
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    sharpe = float("nan")
    sortino = float("nan")
    cagr = 0.0
    calmar = float("nan")

    if not equity_df.empty:
        equity_df["time"] = pd.to_datetime(equity_df["time"])
        equity_df.sort_values("time", inplace=True)
        equity_df["dd"] = equity_df["equity"].cummax() - equity_df["equity"]
        max_drawdown = float(equity_df["dd"].max())
        max_equity = float(equity_df["equity"].cummax().max())
        if max_equity > 0:
            max_drawdown_pct = (max_drawdown / max_equity) * 100

        # Daily returns based metrics
        eq_ts = equity_df.set_index("time")["equity"].asfreq(None)
        daily_eq = eq_ts.resample("1D").last().ffill()
        daily_ret = daily_eq.pct_change().dropna()
        sharpe = _safe_sharpe(daily_ret)
        sortino = _safe_sortino(daily_ret)
        cagr = _cagr(daily_eq)
        calmar = (
            (cagr / (max_drawdown_pct / 100)) if max_drawdown_pct > 0 else float("inf")
        )

    # Streak stats
    margin_calls = (
        int(len(history_df[history_df["exit_reason"] == "Margin Call"]))
        if "exit_reason" in history_df.columns
        else 0
    )
    max_concurrent = max(getattr(self, "concurrent_trades_log", []) or [0])

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
        "margin_calls": int(margin_calls),
        "max_concurrent_trades": int(max_concurrent),
        "avg_trade_size": (
            float(history_df["lot_size"].mean())
            if "lot_size" in history_df.columns
            else float("nan")
        ),
        "min_trade_size": (
            float(history_df["lot_size"].min())
            if "lot_size" in history_df.columns
            else float("nan")
        ),
        "max_trade_size": (
            float(history_df["lot_size"].max())
            if "lot_size" in history_df.columns
            else float("nan")
        ),
        # Streak stats (fall back to 0 if not tracked)
        "max_consecutive_losses": int(getattr(self, "max_consecutive_losses", 0)),
        "max_consecutive_loss_amount": float(
            getattr(self, "max_consecutive_loss_amount", 0.0)
        ),
        "max_consecutive_wins": int(getattr(self, "max_consecutive_wins", 0)),
        "max_consecutive_win_amount": float(
            getattr(self, "max_consecutive_win_amount", 0.0)
        ),
        # Time stats
        "avg_trade_duration": (
            history_df["duration"].mean()
            if "duration" in history_df.columns
            else pd.Timedelta(0)
        ),
        "median_trade_duration": (
            history_df["duration"].median()
            if "duration" in history_df.columns
            else pd.Timedelta(0)
        ),
        "min_trade_duration": (
            history_df["duration"].min()
            if "duration" in history_df.columns
            else pd.Timedelta(0)
        ),
        "max_trade_duration": (
            history_df["duration"].max()
            if "duration" in history_df.columns
            else pd.Timedelta(0)
        ),
        # Added risk/return metrics
        "daily_sharpe": sharpe,  # Above 2 is good enough
        "daily_sortino": sortino,  # Above 2 is good
        "cagr": cagr,  # higher = better
        "calmar": calmar,  # Aim for a range between 2 and 5.0
        # Distribution extras
        "median_trade_pnl": (
            float(history_df["pnl"].median()) if "pnl" in history_df.columns else 0.0
        ),
        "pnl_std": (
            float(history_df["pnl"].std())
            if "pnl" in history_df.columns
            else float("nan")
        ),
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
    print(f"Margin Calls: {report['margin_calls']}")
    print(
        f"Max Consecutive Losses: {report['max_consecutive_losses']} "
        f"(Total Loss: ${abs(report['max_consecutive_loss_amount']):.2f})"
    )
    print(
        f"Max Consecutive Wins: {report['max_consecutive_wins']} "
        f"(Total Profit: ${report['max_consecutive_win_amount']:.2f})"
    )
    print(f"Max Concurrent Trades: {report['max_concurrent_trades']}")
    print("-" * 35)
    print("--- Trade Duration ---")
    print(f"Average Duration: {report['avg_trade_duration']}")
    print(f"Median Duration: {report['median_trade_duration']}")
    print(f"Shortest Duration: {report['min_trade_duration']}")
    print(f"Longest Duration: {report['max_trade_duration']}")
    print("=" * 50 + "\n")


def plot_equity_curve(self, report: Dict[str, Any]):
    strategy_name = report["strategy_name"]
    filename = (
        f"{strategy_name.replace(' ', '_').replace('/', '_').lower()}_equity_curve.png"
    )
    filepath = Path("results") / filename
    equity_df = pd.DataFrame(self.equity_curve)
    if equity_df.empty:
        logger.warning(f"Cannot plot equity curve for {strategy_name}: No equity data.")
        return
    equity_df["time"] = pd.to_datetime(equity_df["time"])
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df["time"], equity_df["equity"], label="Equity")
    plt.title(f"Equity Curve for: {strategy_name}")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.tight_layout()
    filepath.parent.mkdir(exist_ok=True)
    plt.savefig(filepath)
    plt.close()
    logger.info(f"Equity curve plot saved to {filepath}")
