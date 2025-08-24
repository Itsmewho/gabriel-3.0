# Account Management Quick Guide

## What this layer does

- Split total equity across strategies (Allocator).
- Size positions per strategy (Sizer): Fixed % risk or Half-Kelly (capped).
- Enforce risk rules (Governor): max concurrent trades, pause on drawdown.
- Keep per-strategy equity ledger + CSV audit.

## Typical flow (inside your main loop)

1. Compute total balance (from broker).
2. Get per-strategy budgets:
   ```py
   alloc = PortfolioAllocator({"ICT": 0.6, "MIDNIGHT": 0.4})
   ict_budget = alloc.budget_for(broker.balance, "ICT")
   ```

### usage :

Build sizer with strategy config:

ict_cfg = StrategyConfig(risk_mode="fixed", risk_pct=0.01,
lot_min=0.01, lot_step=0.01, lot_max=5)
sizer = AccountSizer(ict_cfg)
req = SizeRequest(balance=ict_budget, sl_pips=50,
value_per_pip=broker.cfg.CONTRACT_SIZE \* 0.0001) # per 1.0 lot
sized = sizer.size(req) # -> lots, risk$

Governor gate:

gov = RiskGovernor({"ICT": ict_cfg})
gov.on_equity("ICT", ledger.equity("ICT"))
if gov.allow_new_trade("ICT").ok and sized.lots >= ict_cfg.lot_min:
tr = broker.open_trade("buy", price, wanted_lots=sized.lots,
sl_pips=50, tp_pips=100, t=ts)
if tr: ledger.on_open("ICT", ts, trade_id=tr.id)

When trade closes (detect via broker events / trade_history at end of day / on close):

# pnl is recorded in trade; broker updates balance. You mirror it into ledger:

ledger.on_close("ICT", tr.exit_time, pnl=tr.pnl, trade_id=tr.id)
gov.on_close("ICT")

Export:

export_account_audit(ledger.snapshot_df(), "results/audit/account_ledger.csv")
