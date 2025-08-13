from threading import Thread
from DWX.python.api.dwx_client import dwx_client


class PrintEvents:
    """Handelt de events van de dwx_client af."""

    def __init__(self, candle_builder_instance):
        self.candle_builder = candle_builder_instance

    def on_tick(self, symbol, bid, ask):
        self.candle_builder.update(symbol, bid)

    def on_bar_data(self, symbol, timeframe, time, *_):
        self.candle_builder.set_last_bar_time(symbol, timeframe, time)

    def on_message(self, msg):
        pass

    def on_order_event(self):
        pass

    def check_open_orders(self):
        pass


def start_dwx_client_thread(candle_builder_instance):
    """Initialiseert en start de DWX client met een specifieke builder instance."""
    print("DWX Client thread wordt gestart...")

    mt5_path = r"C:/Users/Itsme/AppData/Roaming/MetaQuotes/Terminal/73B7A2420D6397DFF9014A20F1201F97/MQL5/Files/"

    handler = PrintEvents(candle_builder_instance)

    # Instantieer de client direct.
    try:
        dwx = dwx_client(event_handler=handler, metatrader_dir_path=mt5_path)
    except Exception as e:
        print(f"[ERROR] Kon de dwx_client niet initialiseren: {e}")
        return  # Stop de thread als de client niet kan starten

    dwx.ACTIVE = True
    dwx.START = True

    for func in [dwx.check_bar_data, dwx.check_market_data]:
        Thread(target=func, daemon=True).start()

    dwx.subscribe_symbols(["EURUSD"])
    dwx.subscribe_symbols_bar_data([["EURUSD", "M1"]])
