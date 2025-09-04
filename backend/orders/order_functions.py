import MetaTrader5 as mt5
from services.mt5_client import connect_mt5
from typing import Optional, Tuple, List, Dict, Union

connect_mt5()


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


def place_market_order(
    symbol: str,
    volume: float,
    order_type: str,  # "buy" | "sell"
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    magic: int = 0,
    comment: Optional[str] = None,  # <-- NEW: e.g., "STRAT=EMA Crossover"
    deviation: int = 10,  # <-- optional override
    filling: Optional[int] = None,  # <-- optional override: e.g., mt5.ORDER_FILLING_IOC
) -> Tuple[bool, str]:
    """
    Sends a market order to MT5.

    Returns:
        (True, "Order placed successfully (ticket=123456)") on success
        (False, "Error message") on failure
    """
    order_map = {"buy": mt5.ORDER_TYPE_BUY, "sell": mt5.ORDER_TYPE_SELL}
    side = (order_type or "").lower()
    if side not in order_map:
        return False, "Invalid order type."

    # Init terminal
    if not mt5.initialize():  # type: ignore
        return False, "MT5 initialization failed"

    try:
        # Ensure symbol is selected/visible
        info = mt5.symbol_info(symbol)  # type: ignore
        if info is None or not info.visible:
            if not mt5.symbol_select(symbol, True):  # type: ignore
                return False, f"Symbol {symbol} not available"

        # Optional extra volume validation (you already have is_valid_volume upstream)
        # Normalize volumes to broker step if needed
        # step = getattr(info, "volume_step", 0.01) or 0.01
        # volume = round(volume / step) * step

        # Price + digits handling
        tick = mt5.symbol_info_tick(symbol)  # type: ignore
        if tick is None:
            return False, f"Failed to get market price for {symbol}"

        price = tick.ask if side == "buy" else tick.bid
        digits = getattr(info, "digits", 5) or 5
        price = round(price, digits)

        # Basic SL/TP sanity vs side
        if stop_loss is not None:
            if (side == "buy" and stop_loss >= price) or (
                side == "sell" and stop_loss <= price
            ):
                return False, "Invalid Stop Loss"
            stop_loss = round(float(stop_loss), digits)

        if take_profit is not None:
            if (side == "buy" and take_profit <= price) or (
                side == "sell" and take_profit >= price
            ):
                return False, "Invalid Take Profit"
            take_profit = round(float(take_profit), digits)

        # Fill policy
        type_filling = filling if filling is not None else mt5.ORDER_FILLING_IOC

        req: Dict[str, Union[int, float, str]] = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_map[side],
            "price": price,
            "deviation": int(deviation),
            "magic": int(magic),
            "comment": comment or "MT5 Order",  # <-- strategy tag lands here
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": type_filling,
        }
        if stop_loss is not None:
            req["sl"] = stop_loss
        if take_profit is not None:
            req["tp"] = take_profit

        result = mt5.order_send(req)  # type: ignore

        if result is None:
            code = mt5.last_error()  # type: ignore
            return False, f"Order failed: {code}"

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            # Combine comment + retcode for easier debugging
            return (
                False,
                f"Order failed: {result.retcode} ({getattr(result, 'comment', '')})",
            )

        ticket = getattr(result, "order", None)
        return True, f"Order placed successfully (ticket={ticket})"

    finally:
        mt5.shutdown()  # type: ignore


def place_pending_order(
    symbol: str,
    volume: float,
    order_type: str,
    price: float,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
) -> bool:
    order_types = {
        "buy_limit": mt5.ORDER_TYPE_BUY_LIMIT,
        "sell_limit": mt5.ORDER_TYPE_SELL_LIMIT,
        "buy_stop": mt5.ORDER_TYPE_BUY_STOP,
        "sell_stop": mt5.ORDER_TYPE_SELL_STOP,
    }

    if order_type not in order_types:
        print("Invalid order type.")
        return False

    request: Dict[str, Union[int, float, str]] = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": volume,
        "type": order_types[order_type],
        "price": price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": 10,
        "magic": 0,
        "comment": "MT5 Pending Order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)  # type: ignore
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed: {result.comment}")
        return False

    print(f"Pending Order Placed: {order_type.upper()} {volume} {symbol} at {price}")
    return True


def check_open_orders(
    magic: Optional[int] = None,
) -> List[Dict[str, Union[int, float, str]]]:
    """
    Haalt openstaande orders op. Als een magic number is opgegeven,
    wordt er gefilterd op alleen de orders van die specifieke strategie.
    """
    if not mt5.initialize():  # type: ignore
        return []

    if magic is None:
        orders = mt5.positions_get()  # type: ignore
    else:
        orders = mt5.positions_get(magic=magic)  # type: ignore

    mt5.shutdown()  # type: ignore

    if not orders:
        return []

    result: List[Dict[str, Union[int, float, str]]] = []
    for order in orders:
        result.append(
            {
                "ticket": order.ticket,
                "symbol": order.symbol,
                "volume": order.volume,
                "price_open": order.price_open,
                "type": order.type,
                "sl": order.sl,
                "tp": order.tp,
                "magic": order.magic,
                "comment": order.comment,
                "profit": getattr(order, "profit", 0.0),
            }
        )
    return result


def close_order(order_id: int) -> bool:
    if not mt5.initialize():  # type: ignore
        print("MT5 initialization failed.")
        return False

    order = mt5.positions_get(ticket=order_id)  # type: ignore
    if not order:
        print(f"Order {order_id} not found.")
        mt5.shutdown()  # type: ignore
        return False

    order = order[0]
    tick = mt5.symbol_info_tick(order.symbol)  # type: ignore
    if tick is None:
        print(f"Failed to get tick for {order.symbol}")
        mt5.shutdown()  # type: ignore
        return False

    price = tick.bid if order.type == mt5.ORDER_TYPE_BUY else tick.ask
    order_type = (
        mt5.ORDER_TYPE_SELL if order.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    )

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": order.symbol,
        "volume": order.volume,
        "type": order_type,
        "price": price,
        "position": order.ticket,
        "deviation": 10,
        "magic": 0,
        "comment": "MT5 Order Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)  # type: ignore
    mt5.shutdown()  # type: ignore

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order close failed: {result.comment}")
        return False

    print(f"Order {order_id} closed at {price}")
    return True


def close_all_orders() -> bool:
    if not mt5.initialize():  # type: ignore
        print("MT5 initialization failed.")
        return False

    positions = mt5.positions_get()  # type: ignore
    if not positions:
        print("No open orders found to close.")
        mt5.shutdown()  # type: ignore
        return True

    all_success = True
    for pos in positions:
        print(f"Closing Order ID: {pos.ticket} | {pos.symbol} | Volume: {pos.volume}")
        success = close_order(pos.ticket)
        if not success:
            all_success = False

    mt5.shutdown()  # type: ignore
    return all_success


def modify_order(
    order_id: int,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
) -> Tuple[bool, str]:
    if not mt5.initialize():  # type: ignore
        return False, "MT5 initialization failed"

    order = mt5.positions_get(ticket=order_id)  # type: ignore
    if not order:
        return False, "Order not found"

    order = order[0]
    symbol = order.symbol
    is_buy = order.type == mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(symbol)  # type: ignore

    if not tick:
        mt5.shutdown()  # type: ignore
        return False, f"Could not fetch current price for {symbol}"

    current_price = tick.ask if is_buy else tick.bid

    if stop_loss:
        if (is_buy and stop_loss >= current_price) or (
            not is_buy and stop_loss <= current_price
        ):
            mt5.shutdown()  # type: ignore
            return False, "Invalid Stop Loss"

    if take_profit:
        if (is_buy and take_profit <= current_price) or (
            not is_buy and take_profit >= current_price
        ):
            mt5.shutdown()  # type: ignore
            return False, "Invalid Take Profit"

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": order_id,
        "sl": stop_loss or 0.0,
        "tp": take_profit or 0.0,
        "magic": 0,
        "comment": "SL/TP Modify",
    }

    result = mt5.order_send(request)  # type: ignore
    mt5.shutdown()  # type: ignore

    if not result or result.retcode != mt5.TRADE_RETCODE_DONE:
        return (
            False,
            f"Modification failed: {result.comment if result else 'Unknown error'}",
        )

    return True, "Order modified successfully"


def get_mt5_leverage() -> Tuple[Optional[int], Optional[str]]:
    if not mt5.initialize():  # type: ignore
        return None, "MT5 initialization failed"

    account_info = mt5.account_info()  # type: ignore
    mt5.shutdown()  # type: ignore

    if account_info is None:
        return None, "Failed to fetch account info"

    return account_info.leverage, None
