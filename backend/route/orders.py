import MetaTrader5 as mt5
from flask import Blueprint, request, jsonify
from typing import Optional, Dict, Any

from services.mt5_client import connect_mt5
from orders.order_functions import (
    place_market_order,
    place_pending_order,
    check_open_orders,
    close_order,
    close_all_orders,
    modify_order,
)


orders_bp = Blueprint("orders", __name__)

connect_mt5()


@orders_bp.route("/api/place_market", methods=["POST"])
def api_place_market_order():
    data: Dict[str, Any] = request.json or {}

    symbol: Optional[str] = data.get("symbol")
    volume: Optional[float] = data.get("volume")
    order_type: Optional[str] = data.get("order_type")
    stop_loss: Optional[float] = data.get("stop_loss")
    take_profit: Optional[float] = data.get("take_profit")

    if not symbol or volume is None or order_type not in ["buy", "sell"]:

        return jsonify({"success": False, "error": "Invalid request data"}), 400

    success, message = place_market_order(
        symbol, volume, order_type, stop_loss, take_profit
    )
    return jsonify({"success": success, "message": message}), (200 if success else 400)


@orders_bp.route("/api/place_pending", methods=["POST"])
def api_place_pending_order():
    data: Dict[str, Any] = request.json or {}
    symbol: Optional[str] = data.get("symbol")
    volume: Optional[float] = data.get("volume")
    order_type: Optional[str] = data.get("order_type")
    price: Optional[float] = data.get("price")
    stop_loss: Optional[float] = data.get("stop_loss")
    take_profit: Optional[float] = data.get("take_profit")

    if not all([symbol, volume, price]) or order_type not in [
        "buy_limit",
        "sell_limit",
        "buy_stop",
        "sell_stop",
    ]:
        return jsonify({"error": "Invalid request data"}), 400

    success = place_pending_order(
        symbol, volume, order_type, price, stop_loss, take_profit  # type: ignore
    )
    return jsonify({"success": success})


@orders_bp.route("/api/open_orders", methods=["GET"])
def api_get_open_orders():
    orders = check_open_orders()
    return jsonify({"orders": orders})


@orders_bp.route("/api/close_order/<int:order_id>", methods=["POST"])
def api_close_order(order_id: int):
    result = close_order(order_id)
    return jsonify({"success": result})


@orders_bp.route("/api/close_all", methods=["POST"])
def api_close_all_orders():
    result = close_all_orders()
    return jsonify({"success": result})


@orders_bp.route("/api/modify_order/<int:order_id>", methods=["POST"])
def api_modify_order(order_id: int):
    data: Dict[str, Any] = request.get_json() or {}
    stop_loss: Optional[float] = data.get("stop_loss")
    take_profit: Optional[float] = data.get("take_profit")

    success, message = modify_order(order_id, stop_loss, take_profit)
    return (
        (
            jsonify({"success": True, "message": message}),
            200,
        )
        if success
        else (
            jsonify({"success": False, "error": message}),
            400,
        )
    )


@orders_bp.route("/api/symbol_tick/<symbol>", methods=["GET"])
def get_symbol_tick(symbol: str):
    if not mt5.initialize():  # type: ignore
        return jsonify({"success": False, "error": "MT5 connection failed"})

    symbol = symbol.upper()
    if not mt5.symbol_select(symbol, True):  # type: ignore
        return jsonify({"success": False, "error": f"Failed to select {symbol}"}), 400

    tick = mt5.symbol_info_tick(symbol)  # type: ignore
    info = mt5.symbol_info(symbol)  # type: ignore

    if not tick or not info:
        return jsonify({"success": False, "error": f"No data for {symbol}"}), 404

    return jsonify(
        {
            "success": True,
            "data": {
                "ask": tick.ask,
                "bid": tick.bid,
                "last": tick.last,
                "time": tick.time,
            },
            "info": {
                "volume_min": info.volume_min,
                "volume_max": info.volume_max,
                "volume_step": info.volume_step,
                "trade_stops_level": info.trade_stops_level,
            },
        }
    )


@orders_bp.route("/api/account_info", methods=["GET"])
def get_account_info():
    if not mt5.initialize():  # type: ignore
        return jsonify({"success": False, "error": "MT5 connection failed"}), 500

    account = mt5.account_info()  # type: ignore
    mt5.shutdown()  # type: ignore

    if not account:
        return jsonify({"success": False, "error": "Failed to fetch account info"}), 500

    # Some MT5 builds return a namedtuple with _asdict()
    try:
        ad = account._asdict()  # type: ignore[attr-defined]
    except Exception:
        ad = None

    trade_stops_level = None
    if ad and isinstance(ad, dict):
        trade_stops_level = ad.get("trade_stops_level")
    else:
        trade_stops_level = getattr(account, "trade_stops_level", None)

    return (
        jsonify(
            {
                "success": True,
                "balance": float(account.balance),
                "equity": float(account.equity),
                "margin_free": float(account.margin_free),
                "margin_level": float(getattr(account, "margin_level", 0.0)),
                "margin_so_call": float(getattr(account, "margin_so_call", 0.0)),
                "profit": float(getattr(account, "profit", 0.0)),
                "leverage": int(getattr(account, "leverage", 0)),
                "trade_stops_level": trade_stops_level,
                "open_orders": len(check_open_orders() or []),
            }
        ),
        200,
    )
