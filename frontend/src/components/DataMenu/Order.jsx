import { useState, useEffect, useCallback } from 'react';
import { useTradingMode } from '../../context/TradingmodeContext';
import { useToast } from '../../context/ToastContext';
import { submitOrder } from '../../utils/submitOrder';
import { fetchAccountInfo as fetchInfo, fetchSymbolTick } from '../../api/Orders';

import styles from './styles/Order.module.css';

const Order = () => {
  const selectedSymbol = 'EURUSD';
  const { showToast } = useToast();
  const { mode } = useTradingMode();

  // State
  const [accountInfo, setAccountInfo] = useState(null);
  const [symbolTick, setSymbolTick] = useState(null);
  const [volume, setVolume] = useState(0.1);
  const [orderType, setOrderType] = useState('buy');
  const [slPoints, setSlPoints] = useState('');
  const [tpPoints, setTpPoints] = useState('');
  const [fadeIn, setFadeIn] = useState(false);

  // Symbol-specifieke configuratie
  const [volumeStep, setVolumeStep] = useState(null);
  const [volumeMin, setVolumeMin] = useState(null);
  const [volumeMax, setVolumeMax] = useState(null);
  const [stopLevelPoints, setStopLevelPoints] = useState(0);

  // Geoptimaliseerde data fetch functie
  const fetchOrderData = useCallback(async () => {
    const [infoRes, tickRes] = await Promise.all([
      fetchInfo(mode),
      fetchSymbolTick(selectedSymbol),
    ]);

    if (infoRes.success) {
      setAccountInfo(infoRes);
    } else {
      showToast(infoRes.error || 'Failed to load account info', 'error');
    }

    if (tickRes.success) {
      if (tickRes.data) setSymbolTick(tickRes.data);
      if (tickRes.info) {
        const stopPoints = tickRes.info.trade_stops_level
          ? parseFloat((tickRes.info.trade_stops_level * 0.00001).toFixed(5))
          : 0;
        setVolumeStep(tickRes.info.volume_step);
        setVolumeMin(tickRes.info.volume_min);
        setVolumeMax(tickRes.info.volume_max);
        setStopLevelPoints(stopPoints);
      } else {
        showToast('Missing symbol info in MT5 response.', 'warning');
      }
    } else {
      showToast(tickRes.error || 'Failed to load symbol data.', 'error');
    }
  }, [mode, showToast, selectedSymbol]);

  useEffect(() => {
    setFadeIn(true);

    fetchOrderData();

    const intervalId = setInterval(fetchOrderData, 5000);

    return () => clearInterval(intervalId);
  }, [fetchOrderData]);

  const currentPrice = orderType === 'buy' ? symbolTick?.ask : symbolTick?.bid || null;

  const requiredMargin = () => {
    if (!accountInfo || !currentPrice) return 0;
    return ((volume * 100000 * currentPrice) / (accountInfo.leverage || 100)).toFixed(2);
  };

  const risk = () => {
    if (!currentPrice || !slPoints) return 0;
    const slPrice =
      currentPrice - parseFloat(slPoints) * 0.00001 * (orderType === 'buy' ? 1 : -1);
    return (Math.abs(currentPrice - slPrice) * 100000 * volume).toFixed(2);
  };

  const reward = () => {
    if (!currentPrice || !tpPoints) return 0;
    const tpPrice =
      currentPrice + parseFloat(tpPoints) * 0.00001 * (orderType === 'buy' ? 1 : -1);
    return (Math.abs(tpPrice - currentPrice) * 100000 * volume).toFixed(2);
  };

  const rrRatio = () => {
    const r = parseFloat(risk());
    const w = parseFloat(reward());
    return r === 0 ? 'N/A' : (w / r).toFixed(2);
  };

  if (!symbolTick || !volumeStep) {
    return <div className={styles.view}>Loading trading info...</div>;
  }

  return (
    <div className={`${styles.view} ${fadeIn ? styles.fadeIn : ''}`}>
      <div className={styles.orderContainer}>
        <h3 className={styles.title}>
          {mode === 'paper' ? 'Paper Order' : 'Live Order'}
        </h3>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            const parsedSL = slPoints !== '' ? parseFloat(slPoints) : null;
            const parsedTP = tpPoints !== '' ? parseFloat(tpPoints) : null;
            submitOrder({
              e,
              selectedSymbol,
              volume,
              slPoints: parsedSL,
              tpPoints: parsedTP,
              orderType,
              currentPrice,
              accountInfo,
              mode,
              volumeMin,
              volumeMax,
              volumeStep,
              stopLevelPoints,
              showToast,
              fetchAccountInfo: fetchOrderData,
              setVolume,
              setSlPoints,
              setTpPoints,
            });
          }}
          style={{ display: 'grid', gap: '1rem' }}
        >
          <p className={styles.subsection}>
            <strong>Symbol:</strong> {selectedSymbol} <strong>Current Price:</strong>{' '}
            {currentPrice}
            {accountInfo && typeof accountInfo.balance === 'number' && (
              <>
                <strong>{mode === 'paper' ? 'Paper Balance' : 'Balance'}:</strong> $
                {accountInfo.balance.toFixed(2)} <strong>Free Margin:</strong> $
                {(mode === 'paper'
                  ? (accountInfo.free_margin ?? 0)
                  : (accountInfo.margin_free ?? 0)
                ).toFixed(2)}
              </>
            )}
          </p>

          {/* Type */}
          <label className={styles.grid}>
            Type:
            <select
              value={orderType}
              className={styles.input_type}
              onChange={(e) => setOrderType(e.target.value)}
            >
              <option value="buy">Buy</option>
              <option value="sell">Sell</option>
            </select>
          </label>

          {/* Volume */}
          <label className={styles.grid}>
            Volume:
            <input
              type="number"
              step={volumeStep}
              min={volumeMin}
              max={volumeMax}
              value={volume}
              className={styles.input_type}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                const precision = volumeStep.toString().split('.')[1]?.length || 2;
                const rounded = parseFloat(val.toFixed(precision));
                if (!isNaN(rounded)) setVolume(rounded);
              }}
            />
          </label>

          {/* SL */}
          <label className={styles.grid}>
            Stop Loss:
            <input
              type="number"
              step="0.0001"
              value={
                currentPrice && slPoints !== ''
                  ? (
                      currentPrice +
                      parseFloat(slPoints) * 0.00001 * (orderType === 'sell' ? 1 : -1)
                    ).toFixed(5)
                  : currentPrice?.toFixed(5) || ''
              }
              className={styles.input_type}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                if (!isNaN(val) && currentPrice) {
                  const rawPoints = (val - currentPrice) / 0.00001;
                  const signedPoints = orderType === 'sell' ? rawPoints : -rawPoints;
                  setSlPoints(Math.abs(signedPoints.toFixed(0)));
                }
              }}
            />
          </label>

          {/* TP */}
          <label className={styles.grid}>
            Take Profit:
            <input
              type="number"
              step="0.0001"
              value={
                currentPrice && tpPoints !== ''
                  ? (
                      currentPrice +
                      parseFloat(tpPoints) * 0.00001 * (orderType === 'sell' ? -1 : 1)
                    ).toFixed(5)
                  : currentPrice?.toFixed(5) || ''
              }
              className={styles.input_type}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                if (!isNaN(val) && currentPrice) {
                  const rawPoints = (val - currentPrice) / 0.00001;
                  const signedPoints = orderType === 'sell' ? -rawPoints : rawPoints;
                  setTpPoints(Math.abs(signedPoints.toFixed(0)));
                }
              }}
            />
          </label>

          {/* R:R */}
          <div className={styles.metrics}>
            <p>
              Required Margin:{' '}
              <span className={styles.color_text}>${requiredMargin()}</span>
            </p>
            <p>
              Risk: <span className={styles.color_text}>${risk()}</span>
            </p>
            <p>
              Reward: <span className={styles.color_text}>${reward()}</span>
            </p>
            <p>
              Risk/Reward: <span className={styles.color_text}>{rrRatio()} : 1</span>
            </p>
          </div>

          <button type="submit" className="btn">
            Submit Order
          </button>
        </form>
      </div>
    </div>
  );
};

export default Order;
