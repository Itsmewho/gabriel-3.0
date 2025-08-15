import time
import re
from datetime import datetime, timedelta
from DWX.python.api.dwx_client import dwx_client

# --- time helpers ---
_TIME_KEY_RE = re.compile(r"^\d{4}[.\-]\d{2}[.\-]\d{2}[ T]\d{2}:\d{2}")


def parse_bar_time_local(s: str) -> datetime:
    """Parse DWX bar time as local/server wall clock (no UTC)."""
    s = (s or "").rstrip("Z").replace("-", ".").replace("T", " ")
    return datetime.strptime(s, "%Y.%m.%d %H:%M")


def _parse_key_local(s: str) -> datetime:
    s = s.replace("-", ".").replace("T", " ")
    return datetime.strptime(s, "%Y.%m.%d %H:%M")


def _fmt_local(dt: datetime) -> str:
    return dt.strftime("%Y.%m.%d %H:%M")


def to_epoch_local(dt: datetime) -> int:
    """Naive local datetime -> epoch seconds using local rules (no UTC)."""
    import time as _t

    return int(_t.mktime(dt.timetuple()))


# --- historic payload normalization ---


def _pick(d, *names):
    for n in names:
        if n in d:
            return d[n]
    return None


def _normalize_bars(payload, anchor_dt: datetime | None = None):
    """
    Normalize DWX historic payload to a list of bars [{time,open,high,low,close,tick_volume}].
    If anchor_dt is provided, shift all bar times by whole-hour offset so last bar aligns.
    """
    bars = []

    # dict keyed by time strings: {"YYYY.MM.DD HH:MM": {open/high/...}, ...}
    if (
        isinstance(payload, dict)
        and payload
        and all(isinstance(k, str) and _TIME_KEY_RE.match(k) for k in payload.keys())
    ):
        keyed = sorted(payload.items(), key=lambda kv: kv[0])
        dts = []
        for k, v in keyed:
            dt = _parse_key_local(k)
            dts.append(dt)
            bars.append(
                {
                    "time": _fmt_local(dt),
                    "open": _pick(v, "open", "Open", "o", "O"),
                    "high": _pick(v, "high", "High", "h", "H"),
                    "low": _pick(v, "low", "Low", "l", "L"),
                    "close": _pick(v, "close", "Close", "c", "C"),
                    "tick_volume": _pick(
                        v, "tick_volume", "TickVolume", "volume", "Volume"
                    ),
                }
            )
        # align times if needed
        if anchor_dt and dts:
            expected_last = anchor_dt.replace(second=0, microsecond=0)
            got_last = dts[-1]
            diff_min = round((expected_last - got_last).total_seconds() / 60)
            hours = int(round(diff_min / 60))  # nearest whole hour
            shift_min = hours * 60
            if hours != 0 and abs(shift_min) <= 12 * 60:
                shift = timedelta(minutes=shift_min)
                # optional: print(f"[HIST-SHIFT] applying {hours:+d}h shift")
                for b in bars:
                    shifted = _parse_key_local(b["time"]) + shift
                    b["time"] = _fmt_local(shifted)
        return bars

    # numeric-key dict: {"0": {...}, "1": {...}}
    if isinstance(payload, dict) and payload and all(str(k).isdigit() for k in payload):
        keys = sorted(payload.keys(), key=lambda x: int(x))
        return [payload[k] for k in keys]

    # dict of arrays: {time:[], open:[], ...}
    if isinstance(payload, dict):
        low = {str(k).lower(): v for k, v in payload.items()}
        req = ("time", "open", "high", "low", "close")
        if all(k in low and isinstance(low[k], list) for k in req):
            n = min(*(len(low[k]) for k in req))
            tv = low.get("tick_volume", [])
            bars = []
            dts = []
            for i in range(n):
                t = low["time"][i]
                if isinstance(t, (int, float)):
                    dt = datetime.fromtimestamp(int(t))
                else:
                    dt = _parse_key_local(str(t))
                dts.append(dt)
                bars.append(
                    {
                        "time": _fmt_local(dt),
                        "open": low["open"][i],
                        "high": low["high"][i],
                        "low": low["low"][i],
                        "close": low["close"][i],
                        "tick_volume": (
                            tv[i] if isinstance(tv, list) and i < len(tv) else None
                        ),
                    }
                )
            if anchor_dt and dts:
                expected_last = anchor_dt.replace(second=0, microsecond=0)
                got_last = dts[-1]
                diff_min = round((expected_last - got_last).total_seconds() / 60)
                if diff_min % 60 == 0 and abs(diff_min) <= 12 * 60:
                    shift = timedelta(minutes=diff_min)
                    for b in bars:
                        shifted = _parse_key_local(b["time"]) + shift
                        b["time"] = _fmt_local(shifted)
            return bars

    # already a list
    if isinstance(payload, list):
        return payload

    return []


# --- event handler ---


class PrintEvents:
    """Print ticks, bars, account; request 10d history anchored to bar time."""

    def __init__(self, candle_builder=None, symbol="EURUSD", timeframe="M1"):
        self.candle_builder = candle_builder
        self.client = None
        self.symbol = symbol
        self.timeframe = timeframe

        self._last_bar_seen = {}  # {(symbol, tf): time_str}
        self._last_account = None  # dict snapshot
        self._last_tick = {}  # {symbol: (bid, ask)}
        self._history_requested = False
        self._last_hist_anchor = None

    def _print_account(self, tag="account"):
        if not self.client or not self.client.account_info:
            return
        ai = self.client.account_info
        snap = {
            "balance": ai.get("balance"),
            "equity": ai.get("equity"),
            "margin_free": ai.get("free_margin") or ai.get("margin_free"),
            "leverage": ai.get("leverage"),
        }
        if snap != self._last_account:
            self._last_account = dict(snap)
            print(
                f"[{tag.upper()}] balance={snap['balance']} equity={snap['equity']} free={snap['margin_free']} lev={snap['leverage']}"
            )

    def _request_10d_history_from_bar(self, time_str: str):
        anchor = parse_bar_time_local(time_str).replace(second=0, microsecond=0)
        self._last_hist_anchor = anchor
        start_dt = anchor - timedelta(days=10)
        end_dt = anchor + timedelta(minutes=0)
        print(
            f"[HIST-REQ] {self.symbol} {self.timeframe} start={start_dt} end={end_dt} (local epoch {to_epoch_local(start_dt)}->{to_epoch_local(end_dt)})"
        )
        self.client.get_historic_data(  # type: ignore
            self.symbol,
            self.timeframe,
            to_epoch_local(start_dt),
            to_epoch_local(end_dt),
        )

    # --- DWX hooks ---
    def on_tick(self, symbol, bid, ask):
        last = self._last_tick.get(symbol)
        if last != (bid, ask):
            self._last_tick[symbol] = (bid, ask)
            spread = (ask - bid) if ask is not None and bid is not None else None
            if spread is not None:
                print(f"[TICK] {symbol} bid={bid} ask={ask} spread={spread}")
            else:
                print(f"[TICK] {symbol} bid={bid} ask={ask}")
        if self.candle_builder:
            self.candle_builder.update(symbol, bid)

    def on_bar_data(
        self, symbol, timeframe, time_str, open_, high, low, close, tick_volume
    ):
        key = (symbol, timeframe)
        if self._last_bar_seen.get(key) == time_str:
            return
        self._last_bar_seen[key] = time_str
        print(
            f"[BAR] {symbol} {timeframe} {time_str} O:{open_} H:{high} L:{low} C:{close} TV:{tick_volume}"
        )
        if self.candle_builder:
            self.candle_builder.set_last_bar_time(symbol, timeframe, time_str)
        if (
            symbol == self.symbol
            and timeframe == self.timeframe
            and not self._history_requested
        ):
            self._history_requested = True
            self._request_10d_history_from_bar(time_str)

    def on_historic_data(self, symbol, timeframe, data):
        bars = _normalize_bars(data, anchor_dt=self._last_hist_anchor)
        if not bars:
            print(
                f"[HIST] {symbol} {timeframe}: keys={list(data.keys())[:8] if isinstance(data, dict) else type(data).__name__}"
            )
            return
        first, last = bars[0], bars[-1]
        print(f"[HIST COUNT] {symbol} {timeframe}: {len(bars)}")
        print(
            f"[HIST FIRST] {symbol} {timeframe} {first.get('time')} O:{first.get('open')} H:{first.get('high')} L:{first.get('low')} C:{first.get('close')}"
        )
        print(
            f"[HIST LAST]  {symbol} {timeframe} {last.get('time')} O:{last.get('open')} H:{last.get('high')} L:{last.get('low')} C:{last.get('close')}"
        )

    def on_order_event(self):
        self._print_account("account_update")

    def on_message(self, msg):
        pass


# --- bootstrap ---


def start_dwx_client(candle_builder=None, symbol="EURUSD", timeframe="M1"):
    mt5_files_dir = r"C:/Users/Itsme/AppData/Roaming/MetaQuotes/Terminal/73B7A2420D6397DFF9014A20F1201F97/MQL5/Files/"
    handler = PrintEvents(candle_builder, symbol, timeframe)
    dwx = dwx_client(
        event_handler=handler, metatrader_dir_path=mt5_files_dir, verbose=True
    )
    handler.client = dwx  # type: ignore

    dwx.ACTIVE = True
    dwx.START = True
    dwx.subscribe_symbols([symbol])
    dwx.subscribe_symbols_bar_data([[symbol, timeframe]])

    # Print initial account info once available
    for _ in range(50):  # ~5s
        if dwx.account_info:
            handler._print_account("initial")
            break
        time.sleep(0.1)

    return dwx


if __name__ == "__main__":
    dwx = start_dwx_client(symbol="EURUSD", timeframe="M1")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        dwx.ACTIVE = False  # type: ignore
        print("Stopped.")
