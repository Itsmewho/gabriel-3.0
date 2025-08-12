import MetaTrader5 as mt5
from flask import Blueprint, request, jsonify
from services.mt5_client import connect_mt5
from operations.sql_operations import insert_record, fetch_records, update_records
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any

paper_bp = Blueprint("paper", __name__)

connect_mt5()


def get_execution_price(
    symbol: str, order_type: str
) -> Tuple[Optional[float], Optional[str]]:
    if not mt5.initialize():  # type: ignore
        return None, "MT5 initialization failed."

    tick = mt5.symbol_info_tick(symbol)  # type: ignore
    mt5.shutdown()  # type: ignore

    if not tick:
        return None, f"Failed to get tick for {symbol}"

    return tick.ask if order_type == "buy" else tick.bid, None


def is_valid_volume(symbol: str, volume: float) -> Tuple[bool, Optional[str]]:
    info = mt5.symbol_info(symbol)  # type: ignore
    if not info:
        return False, "Symbol not found"

    volume_step = info.volume_step
    volume_min = info.volume_min
    volume_max = info.volume_max

    if not (volume_min <= volume <= volume_max):
        return False, f"Volume must be between {volume_min} and {volume_max}"

    steps = round((volume - volume_min) / volume_step)
    expected = volume_min + steps * volume_step

    if round(expected, 2) != round(volume, 2):
        return False, f"Volume must align with step of {volume_step}"

    return True, None


@paper_bp.route("/api/paper/place", methods=["POST"])
def place_paper_order():
    data: Dict[str, Any] = request.get_json()
    symbol = data.get("symbol")
    volume = data.get("volume")
    order_type = data.get("order_type")
    stop_loss = data.get("stop_loss")
    take_profit = data.get("take_profit")

    if not all([symbol, volume, order_type]):
        return jsonify({"success": False, "error": "Missing required fields"}), 400

    is_valid, volume_msg = is_valid_volume(symbol, volume)  # type: ignore
    if not is_valid:
        return jsonify({"success": False, "error": volume_msg}), 400

    entry_price, error = get_execution_price(symbol, order_type)  # type: ignore
    if error:
        return jsonify({"success": False, "error": error}), 500

    account = fetch_records("paper_account", limit=1)
    if not account:
        return jsonify({"success": False, "error": "Paper account not found"}), 500
    account = account[0]

    leverage = account["leverage"]
    required_margin = (volume * 100000 * entry_price) / leverage  # type: ignore

    if required_margin > account["free_margin"]:
        return jsonify({"success": False, "error": "Not enough free margin"}), 400

    now = datetime.now(timezone.utc)
    insert_record(
        "paper_trades",
        columns=[
            "symbol",
            "type",
            "volume",
            "entry_price",
            "stop_loss",
            "take_profit",
            "entry_time",
            "status",
        ],
        values=(
            symbol,
            order_type,
            volume,
            entry_price,
            stop_loss,
            take_profit,
            now,
            "open",
        ),
    )

    update_records(
        "paper_account",
        set_clause="margin = %s, free_margin = %s, last_updated = %s",
        where_clause="id = %s",
        params=(
            account["margin"] + required_margin,
            account["balance"] - (account["margin"] + required_margin),
            now,
            account["id"],
        ),
    )

    return jsonify({"success": True, "message": "Paper order placed!"})


@paper_bp.route("/api/paper/open", methods=["GET"])
def get_open_paper_trades():
    try:
        trades = fetch_records(
            "paper_trades", where_clause="status = %s", params=("open",)
        )
        formatted = [
            {
                "ticket": t["id"],
                "symbol": t["symbol"],
                "volume": t["volume"],
                "price_open": t["entry_price"],
                "type": 0 if t["type"] == "buy" else 1,
                "sl": t["stop_loss"],
                "tp": t["take_profit"],
            }
            for t in trades
        ]
        return jsonify({"success": True, "orders": formatted})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@paper_bp.route("/api/paper/evaluate", methods=["POST"])
def evaluate_paper_trades():
    try:
        open_trades = fetch_records("paper_trades", where_clause="status = 'open'")
        if not open_trades:
            return jsonify({"success": True, "message": "No open paper trades."})

        if not mt5.initialize():  # type: ignore
            return (
                jsonify({"success": False, "error": "MT5 initialization failed"}),
                500,
            )

        closed_count = 0

        for trade in open_trades:
            symbol, order_type, entry, volume = (
                trade["symbol"],
                trade["type"],
                trade["entry_price"],
                trade["volume"],
            )
            sl, tp = trade["stop_loss"], trade["take_profit"]
            tick = mt5.symbol_info_tick(symbol)  # type: ignore
            if not tick:
                continue

            current = tick.bid if order_type == "sell" else tick.ask
            should_close = (
                order_type == "buy"
                and ((sl and current <= sl) or (tp and current >= tp))
            ) or (
                order_type == "sell"
                and ((sl and current >= sl) or (tp and current <= tp))
            )

            if should_close:
                pnl = (
                    (current - entry) * 100000 * volume
                    if order_type == "buy"
                    else (entry - current) * 100000 * volume
                )
                pnl_pct = (
                    ((current - entry) / entry) * 100
                    if order_type == "buy"
                    else ((entry - current) / entry) * 100
                )
                now = datetime.now(timezone.utc)

                update_records(
                    "paper_trades",
                    set_clause="status = 'closed', exit_price = %s, exit_time = %s, pnl_percent = %s, pnl_absolute = %s",
                    where_clause="id = %s",
                    params=(
                        current,
                        now,
                        round(pnl_pct, 2),
                        round(pnl, 2),
                        trade["id"],
                    ),
                )

                insert_record(
                    "trade_history",
                    columns=[
                        "symbol",
                        "type",
                        "volume",
                        "entry_price",
                        "exit_price",
                        "pnl_percent",
                        "pnl_absolute",
                        "timestamp",
                        "mode",
                    ],
                    values=(
                        symbol,
                        order_type,
                        volume,
                        entry,
                        current,
                        round(pnl_pct, 2),
                        round(pnl, 2),
                        now,
                        "paper",
                    ),
                )

                closed_count += 1

        mt5.shutdown()  # type: ignore
        return jsonify({"success": True, "closed": closed_count})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@paper_bp.route("/api/paper/close/<int:trade_id>", methods=["POST"])
def close_paper_trade(trade_id: int):
    try:
        records = fetch_records(
            "paper_trades",
            where_clause="id = %s AND status = 'open'",
            params=(trade_id,),
            limit=1,
        )
        if not records:
            return (
                jsonify(
                    {"success": False, "error": "Trade not found or already closed"}
                ),
                404,
            )

        trade = records[0]
        symbol, order_type, entry_price, volume = (
            trade["symbol"],
            trade["type"],
            trade["entry_price"],
            trade["volume"],
        )
        price, error = get_execution_price(symbol, order_type)
        if error:
            return jsonify({"success": False, "error": error}), 500

        pnl_absolute = (
            (price - entry_price) * 100000 * volume
            if order_type == "buy"
            else (entry_price - price) * 100000 * volume
        )
        pnl_percent = (
            ((price - entry_price) / entry_price) * 100
            if order_type == "buy"
            else ((entry_price - price) / entry_price) * 100
        )
        now = datetime.now(timezone.utc)

        update_records(
            "paper_trades",
            set_clause="status = 'closed', exit_price = %s, exit_time = %s, pnl_percent = %s, pnl_absolute = %s",
            where_clause="id = %s",
            params=(
                round(price, 5),  # type: ignore
                now,
                round(pnl_percent, 2),
                round(pnl_absolute, 2),
                trade_id,
            ),
        )

        insert_record(
            "trade_history",
            columns=[
                "symbol",
                "type",
                "volume",
                "entry_price",
                "exit_price",
                "pnl_percent",
                "pnl_absolute",
                "timestamp",
                "mode",
            ],
            values=(
                symbol,
                order_type,
                volume,
                entry_price,
                round(price, 5),  # type: ignore
                round(pnl_percent, 2),
                round(pnl_absolute, 2),
                now,
                "paper",
            ),
        )

        open_trades = fetch_records("paper_trades", where_clause="status = 'open'")
        account = fetch_records("paper_account", limit=1)[0]
        leverage = account["leverage"]
        total_margin = sum(
            [(t["entry_price"] * t["volume"] * 100000) / leverage for t in open_trades]
        )
        new_balance = account["balance"] + pnl_absolute

        update_records(
            "paper_account",
            set_clause="balance = %s, equity = %s, margin = %s, free_margin = %s, last_updated = %s",
            where_clause="id = %s",
            params=(
                round(new_balance, 2),
                round(new_balance, 2),
                round(total_margin, 2),
                round(new_balance - total_margin, 2),
                now,
                account["id"],
            ),
        )

        return jsonify({"success": True, "message": "Trade closed and account updated"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@paper_bp.route("/api/paper/modify/<int:ticket>", methods=["POST"])
def modify_paper_order(ticket: int):
    data = request.get_json()
    stop_loss = data.get("stop_loss")
    take_profit = data.get("take_profit")

    try:
        record = fetch_records(
            "paper_trades",
            where_clause="id = %s AND status = 'open'",
            params=(ticket,),
            limit=1,
        )
        if not record:
            return (
                jsonify(
                    {"success": False, "message": "Trade not found or already closed"}
                ),
                404,
            )

        trade = record[0]
        symbol = trade["symbol"]
        order_type = trade["type"]
        tick_price, error = get_execution_price(symbol, order_type)
        if error:
            return jsonify({"success": False, "message": error}), 500

        is_buy = order_type == "buy"
        if stop_loss is not None and (
            (is_buy and stop_loss >= tick_price)
            or (not is_buy and stop_loss <= tick_price)
        ):
            return jsonify({"success": False, "message": "Invalid SL"}), 400
        if take_profit is not None and (
            (is_buy and take_profit <= tick_price)
            or (not is_buy and take_profit >= tick_price)
        ):
            return jsonify({"success": False, "message": "Invalid TP"}), 400

        set_clause = []
        params = []

        if stop_loss is not None:
            set_clause.append("stop_loss = %s")
            params.append(stop_loss)
        if take_profit is not None:
            set_clause.append("take_profit = %s")
            params.append(take_profit)
        if not set_clause:
            return jsonify({"success": False, "message": "No updates provided"}), 400

        params.append(ticket)
        updated = update_records(
            "paper_trades",
            set_clause=", ".join(set_clause),
            where_clause="id = %s",
            params=params,
        )
        return jsonify(
            {"success": True, "message": "Order updated"}
            if updated
            else {"success": False, "message": "Update failed"}
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@paper_bp.route("/api/paper/account_info", methods=["GET"])
def get_paper_account():
    try:
        account = fetch_records("paper_account", limit=1)
        if not account:
            insert_record(
                "paper_account",
                columns=["balance", "equity", "margin", "free_margin", "leverage"],
                values=(100000.0, 100000.0, 0.0, 100000.0, 100),
            )
            account = fetch_records("paper_account", limit=1)

        acc = account[0]
        return jsonify(
            {
                "success": True,
                "balance": acc["balance"],
                "equity": acc["equity"],
                "margin_free": acc["free_margin"],
                "margin_level": None,
                "margin_so_call": None,
                "profit": 0.0,
                "leverage": acc["leverage"],
                "trade_stops_level": None,
                "open_orders": 0,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
