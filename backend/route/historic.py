from flask import Blueprint, request, jsonify
from operations.sql_operations import insert_record, fetch_records
from datetime import datetime, timedelta
from typing import List, Dict, Any
import MetaTrader5 as mt5

historic_bp = Blueprint("historic", __name__)


@historic_bp.route("/api/historic", methods=["GET"])
def get_historic_trades():
    mode: str = request.args.get("mode", "live")
    if mode not in ["live", "paper"]:
        return jsonify({"error": "Invalid mode"}), 400

    if mode == "live":
        if not mt5.initialize():  # type: ignore
            return jsonify({"error": "MT5 initialization failed"}), 500
        try:
            from_date: datetime = datetime.now() - timedelta(days=14)
            to_date: datetime = datetime.now() + timedelta(days=2)
            deals = mt5.history_deals_get(from_date, to_date)  # type: ignore
            if not deals:
                return jsonify({"success": True, "trades": []}), 200

            for deal in deals:
                if deal.type not in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL):  # type: ignore
                    continue

                # (optional) ignore zero PnL if you only want realized outcomes
                if float(getattr(deal, "profit", 0) or 0) == 0.0:
                    continue

                ticket: int = deal.ticket
                exists = fetch_records(
                    "trade_history",
                    where_clause="ticket = %s AND mode = %s",
                    params=(ticket, mode),
                    limit=1,
                )
                if exists:
                    continue

                symbol: str = deal.symbol
                trade_type: str = "buy" if deal.type == mt5.DEAL_TYPE_BUY else "sell"  # type: ignore
                volume: float = deal.volume
                entry_price: float = deal.price
                exit_price: float = deal.price
                pnl_absolute: float = round(float(deal.profit), 2)
                timestamp: datetime = datetime.fromtimestamp(deal.time)

                try:
                    direction: int = 1 if trade_type == "buy" else -1
                    pnl_percent: float = round(
                        (pnl_absolute / (volume * 100000 * entry_price))
                        * 100
                        * direction,
                        2,
                    )
                except Exception:
                    pnl_percent = 0.0

                cmt = getattr(deal, "comment", "") or ""
                strategy = (
                    cmt.split("STRAT=", 1)[1].split()[0][:64]
                    if "STRAT=" in cmt
                    else None
                )

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
                        ticket,
                        symbol,
                        trade_type,
                        volume,
                        round(entry_price, 5),
                        round(exit_price, 5),
                        pnl_percent,
                        pnl_absolute,
                        timestamp,
                        mode,
                        strategy,
                    ),
                )
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            mt5.shutdown()  # type: ignore

    try:
        records: List[Dict[str, Any]] = fetch_records(
            "trade_history",
            columns=(
                "ticket, symbol, type, volume, entry_price, exit_price, "
                "pnl_percent, pnl_absolute, timestamp, mode, strategy"
            ),
            where_clause="mode = %s",
            params=(mode,),
        )

        formatted: List[Dict[str, Any]] = []
        for row in records or []:
            formatted.append(
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
            )

        return jsonify({"success": True, "trades": formatted}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
