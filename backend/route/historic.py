from flask import Blueprint, request, jsonify
from operations.sql_operations import insert_record, fetch_records
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import MetaTrader5 as mt5

historic_bp = Blueprint("historic", __name__)


@historic_bp.route("/api/historic", methods=["GET"])
def get_historic_trades():
    mode: str = request.args.get("mode", "live")
    if mode not in ["live", "paper"]:
        return jsonify({"error": "Invalid mode"}), 400

    account: Optional[Dict[str, Any]] = None
    mt5_ok = False

    if mode == "live":
        mt5_ok = bool(mt5.initialize())  # type: ignore
        if mt5_ok:
            try:
                acc = mt5.account_info()  # type: ignore
                if acc:
                    account = {
                        "balance": acc.balance,
                        "equity": acc.equity,
                        "margin_free": acc.margin_free,
                        "leverage": acc.leverage,
                    }

                # fetch direct from MT5
                from_date = datetime.now() - timedelta(days=14)
                to_date = datetime.now() + timedelta(days=2)
                deals = mt5.history_deals_get(from_date, to_date)  # type: ignore

                trades: List[Dict[str, Any]] = []
                if deals:
                    for deal in deals:
                        # only filled buys/sells
                        if deal.type not in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL):  # type: ignore
                            continue
                        # filters
                        symbol = (getattr(deal, "symbol", "") or "").strip()
                        volume = float(getattr(deal, "volume", 0) or 0)
                        if not symbol or volume == 0:
                            continue
                        if float(getattr(deal, "profit", 0) or 0) == 0.0:
                            continue

                        trade_type = "buy" if deal.type == mt5.DEAL_TYPE_BUY else "sell"  # type: ignore
                        entry_price = float(getattr(deal, "price", 0.0) or 0.0)
                        exit_price = entry_price
                        pnl_abs = round(float(deal.profit), 2)
                        ts = datetime.fromtimestamp(deal.time)

                        try:
                            direction = 1 if trade_type == "buy" else -1
                            pnl_pct = round(
                                (pnl_abs / (volume * 100000 * entry_price))
                                * 100
                                * direction,
                                2,
                            )
                        except Exception:
                            pnl_pct = 0.0

                        cmt = getattr(deal, "comment", "") or ""
                        strategy = (
                            cmt.split("STRAT=", 1)[1].split()[0][:64]
                            if "STRAT=" in cmt
                            else None
                        )

                        t = {
                            "ticket": int(deal.ticket),
                            "symbol": symbol,
                            "type": trade_type,
                            "volume": volume,
                            "entry_price": round(entry_price, 5),
                            "exit_price": round(exit_price, 5),
                            "pnl_percent": pnl_pct,
                            "pnl_absolute": pnl_abs,
                            "timestamp": ts,
                            "mode": mode,
                            "strategy": strategy,
                        }
                        trades.append(t)

                        # optional: keep DB in sync without dupes
                        exists = fetch_records(
                            "trade_history",
                            where_clause="ticket = %s AND mode = %s",
                            params=(t["ticket"], mode),
                            limit=1,
                        )
                        if not exists:
                            insert_record(
                                "trade_history",
                                columns=[
                                    "ticket",
                                    "symbol",
                                    "type",
                                    "volume",
                                    "entry_price",
                                    "exit_price",
                                    "pnl_percent",
                                    "pnl_absolute",
                                    "timestamp",
                                    "mode",
                                    "strategy",
                                ],
                                values=(
                                    t["ticket"],
                                    t["symbol"],
                                    t["type"],
                                    t["volume"],
                                    t["entry_price"],
                                    t["exit_price"],
                                    t["pnl_percent"],
                                    t["pnl_absolute"],
                                    t["timestamp"],
                                    t["mode"],
                                    t["strategy"],
                                ),
                            )

                return (
                    jsonify({"success": True, "account": account, "trades": trades}),
                    200,
                )

            except Exception as e:
                return jsonify({"error": str(e)}), 500
            finally:
                mt5.shutdown()  # type: ignore

    # paper mode or live fallback -> read from DB
    try:
        if mode == "paper":
            acct_rows = fetch_records("paper_account", limit=1)
            if not acct_rows:
                insert_record(
                    "paper_account",
                    columns=["balance", "equity", "margin", "free_margin", "leverage"],
                    values=(100000.0, 100000.0, 0.0, 100000.0, 100),
                )
                acct_rows = fetch_records("paper_account", limit=1)
            acc = acct_rows[0]
            account = {
                "balance": acc["balance"],
                "equity": acc["equity"],
                "margin_free": acc["free_margin"],
                "leverage": acc["leverage"],
            }

        records: List[Dict[str, Any]] = fetch_records(
            "trade_history",
            columns=(
                "ticket, symbol, type, volume, entry_price, exit_price, "
                "pnl_percent, pnl_absolute, timestamp, mode, strategy"
            ),
            where_clause="mode = %s AND COALESCE(symbol,'') <> '' AND COALESCE(volume,0) <> 0",
            params=(mode,),
        )

        trades = [
            {
                key: row.get(key)
                for key in [
                    "ticket",
                    "symbol",
                    "type",
                    "volume",
                    "entry_price",
                    "exit_price",
                    "pnl_percent",
                    "pnl_absolute",
                    "timestamp",
                    "mode",
                    "strategy",
                ]
            }
            for row in (records or [])
        ]

        return jsonify({"success": True, "account": account, "trades": trades}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
