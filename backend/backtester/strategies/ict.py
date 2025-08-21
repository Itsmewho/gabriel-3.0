from backtester.strategies.base_strat import BaseStrategy


class ICTStrategy(BaseStrategy):
    def on_bar(self, broker, t, row):
        px = float(row["close"])

        # your ICT entry conditions here
        if some_condition:
            tr = broker.open_trade(
                side="buy",
                price=px,
                wanted_lots=0.5,
                sl_pips=self.config["sl_pips"],
                tp_pips=self.config["tp_pips"],
                t=t,
            )
            if tr:
                self.setup_trade(broker, tr)  # apply BE/TS
