import { useState, useEffect, useCallback } from 'react';
import { useTradingMode } from '../../context/TradingmodeContext';
import { useToast } from '../../context/ToastContext';
import {
  fetchOpenOrders,
  fetchSymbolTick,
  modifyOrder,
  closeOrder,
} from '../../api/Orders';
import styles from './styles/Openorder.module.css';
import { FETCHTIME } from '../../config/constants';

const OpenOrder = () => {
  const trailingDistance = 0.0002;
  const { mode } = useTradingMode();
  const { showToast } = useToast();
  const [openOrders, setOpenOrders] = useState([]);
  const [symbolTick, setSymbolTick] = useState(null);
  const [modifications, setModifications] = useState({});
  const [lastSLNotified, setLastSLNotified] = useState({});

  const loadOrdersAndPrice = useCallback(async () => {
    const ordersRes = await fetchOpenOrders(mode);
    if (ordersRes.success) setOpenOrders(ordersRes.orders);

    const tickRes = await fetchSymbolTick('EURUSD');
    if (tickRes.success) setSymbolTick(tickRes.data);
  }, [mode]);

  useEffect(() => {
    loadOrdersAndPrice();
    const interval = setInterval(loadOrdersAndPrice, FETCHTIME);
    return () => clearInterval(interval);
  }, [loadOrdersAndPrice]);

  const handleInputChange = (ticket, field, value) => {
    setModifications((prev) => ({
      ...prev,
      [ticket]: {
        ...prev[ticket],
        [field]: value,
      },
    }));
  };

  const handleTrailingSL = useCallback(
    async (order, price) => {
      const direction = order.type === 0 ? 'BUY' : 'SELL';
      const newSl =
        direction === 'BUY' ? price - trailingDistance : price + trailingDistance;

      const currentSl = parseFloat(order.sl);
      const entry = parseFloat(order.price_open);
      const lastNotifiedSl = lastSLNotified[order.ticket] ?? 0;

      const shouldUpdate =
        direction === 'BUY'
          ? newSl > currentSl && newSl > entry
          : newSl < currentSl && newSl < entry;

      if (shouldUpdate) {
        const res = await modifyOrder(mode, order.ticket, {
          stop_loss: newSl.toFixed(5),
          take_profit: order.tp,
        });

        if (res.success) {
          const slMovedEnough = Math.abs(newSl - lastNotifiedSl) >= trailingDistance;
          const passedBreakEven =
            direction === 'BUY'
              ? newSl > entry && lastNotifiedSl <= entry
              : newSl < entry && lastNotifiedSl >= entry;

          if (slMovedEnough || passedBreakEven) {
            showToast(
              `Trailing SL updated for ${order.ticket} ${
                passedBreakEven ? '(break-even)' : ''
              }`,
              'info',
            );

            setLastSLNotified((prev) => ({
              ...prev,
              [order.ticket]: newSl,
            }));
          }
        }
      }
    },
    [mode, showToast, lastSLNotified],
  );

  useEffect(() => {
    const autoMonitor = async () => {
      const resOrders = await fetchOpenOrders(mode);
      const resTick = await fetchSymbolTick('EURUSD');

      if (!resOrders.success || !resTick.success) return;

      const orders = resOrders.orders;
      const bid = resTick.data?.bid;
      const ask = resTick.data?.ask;

      if (!bid || !ask) return;

      for (const order of orders) {
        const direction = order.type === 0 ? 'BUY' : 'SELL';
        const price = direction === 'BUY' ? ask : bid;
        await handleTrailingSL(order, price);
      }

      for (const order of orders) {
        const direction = order.type === 0 ? 'BUY' : 'SELL';
        const price = direction === 'BUY' ? ask : bid;

        const slHit =
          order.sl &&
          ((direction === 'BUY' && price <= order.sl) ||
            (direction === 'SELL' && price >= order.sl));

        const tpHit =
          order.tp &&
          ((direction === 'BUY' && price >= order.tp) ||
            (direction === 'SELL' && price <= order.tp));

        if (slHit || tpHit) {
          const result = await closeOrder(mode, order.ticket);
          if (result.success) {
            showToast(
              `Order ${order.ticket} closed (${slHit ? 'SL hit' : 'TP hit'})`,
              'info',
            );
          }
        }
      }
    };

    const interval = setInterval(() => {
      if (mode === 'paper') {
        autoMonitor();
      }
    }, FETCHTIME);

    return () => clearInterval(interval);
  }, [mode, showToast, handleTrailingSL]);

  const handleModify = async (ticket) => {
    const mod = modifications[ticket] || {};
    const original = openOrders.find((o) => o.ticket === ticket);

    const sl = parseFloat(mod.sl ?? original?.sl);
    const tp = parseFloat(mod.tp ?? original?.tp);
    const entry = parseFloat(original?.price_open);
    const type = original?.type; // 0 = BUY, 1 = SELL

    if (isNaN(sl) || isNaN(tp)) {
      showToast('SL and TP are required', 'error');
      return;
    }

    // Directional validation
    if (type === 0 && (sl >= entry || tp <= entry)) {
      showToast('For BUY: SL must be below and TP above entry price.', 'error');
      return;
    }

    if (type === 1 && (sl <= entry || tp >= entry)) {
      showToast('For SELL: SL must be above and TP below entry price.', 'error');
      return;
    }

    const res = await modifyOrder(mode, ticket, {
      stop_loss: sl,
      take_profit: tp,
    });

    if (res.success) {
      showToast('Order modified successfully', 'success');
      loadOrdersAndPrice();
    } else {
      showToast(res.error || 'Modification failed', 'error');
    }
  };

  const handleClose = async (ticket) => {
    const res = await closeOrder(mode, ticket);
    if (res.success) {
      showToast('Order closed successfully', 'success');
      loadOrdersAndPrice();
    } else {
      showToast(res.error || 'Failed to close order', 'error');
    }
  };

  const bid = symbolTick?.bid ?? null;
  const ask = symbolTick?.ask ?? null;

  return (
    <div className={styles.ordersWrapper}>
      <div className={styles.toggleWrapper}>
        <span className={styles.modeLabel}>
          {mode === 'live' ? 'Live Mode' : 'Paper Mode'}
        </span>
      </div>
      {openOrders.length === 0 ? (
        <p>No open orders</p>
      ) : (
        openOrders.map((order) => {
          const { ticket, symbol, type, sl, tp } = order;
          const direction = type === 0 ? 'BUY' : 'SELL';
          const volume = parseFloat(order.volume);
          const priceOpen = parseFloat(order.price_open);
          const current = direction === 'BUY' ? bid : ask;

          let pnl = 0;
          if (!isNaN(volume) && !isNaN(priceOpen) && !isNaN(current)) {
            const diff = direction === 'BUY' ? current - priceOpen : priceOpen - current;
            pnl = diff * 100000 * volume;
          }

          return (
            <div key={ticket} className={styles.orderBox}>
              <p>Ticket: {ticket}</p>
              <p>{symbol}</p>
              <p>{direction}</p>
              <p className={styles.border}>Vol: {volume}</p>
              <p>Entry: {priceOpen.toFixed(5)}</p>
              <p>Bid: {symbolTick ? symbolTick.bid.toFixed(5) : '...'}</p>
              <p className={styles.border}>
                Ask: {symbolTick ? symbolTick.ask.toFixed(5) : '...'}
              </p>
              <p>SL: {sl != null ? parseFloat(sl).toFixed(5) : 'N/A'}</p>
              <p>TP: {tp != null ? parseFloat(tp).toFixed(5) : 'N/A'}</p>
              <p style={{ color: pnl >= 0 ? '#0a8a77' : '#f23645' }}>
                ${pnl.toFixed(2)} USD
              </p>

              <div className={styles.orderControls}>
                <label className={styles.grid}>
                  SL:
                  <input
                    type="number"
                    step="0.0001"
                    className={styles.input_type}
                    value={
                      modifications[ticket]?.sl !== undefined
                        ? modifications[ticket].sl
                        : sl != null
                          ? parseFloat(sl).toFixed(5)
                          : ''
                    }
                    onChange={(e) => {
                      const val = parseFloat(e.target.value);
                      if (!isNaN(val)) {
                        // Prevent wrong SL direction
                        if (
                          (type === 0 && val < priceOpen) || // BUY: SL must be < entry
                          (type === 1 && val > priceOpen) // SELL: SL must be > entry
                        ) {
                          handleInputChange(ticket, 'sl', val.toFixed(5));
                        } else {
                          showToast(
                            'Invalid SL: must be below entry for BUY, above for SELL',
                            'error',
                          );
                        }
                      }
                    }}
                  />
                </label>

                <label className={styles.grid}>
                  TP:
                  <input
                    type="number"
                    step="0.0001"
                    className={styles.input_type}
                    value={
                      modifications[ticket]?.tp !== undefined
                        ? modifications[ticket].tp
                        : tp != null
                          ? parseFloat(tp).toFixed(5)
                          : ''
                    }
                    onChange={(e) => {
                      const val = parseFloat(e.target.value);
                      if (!isNaN(val)) {
                        // Prevent wrong TP direction
                        if (
                          (type === 0 && val > priceOpen) || // BUY: TP must be > entry
                          (type === 1 && val < priceOpen) // SELL: TP must be < entry
                        ) {
                          handleInputChange(ticket, 'tp', val.toFixed(5));
                        } else {
                          showToast(
                            'Invalid TP: must be above entry for BUY, below for SELL',
                            'error',
                          );
                        }
                      }
                    }}
                  />
                </label>

                <button onClick={() => handleModify(ticket)} className="btn">
                  Modify
                </button>
                <button
                  onClick={() => handleClose(ticket)}
                  className="btn"
                  style={{ marginLeft: '0.5rem' }}
                >
                  Close
                </button>
              </div>
            </div>
          );
        })
      )}
    </div>
  );
};

export default OpenOrder;
