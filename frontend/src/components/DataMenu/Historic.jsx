import styles from './styles/Historic.module.css';
import { useEffect, useState } from 'react';
import { useTradingMode } from '../../context/TradingmodeContext';
import { fetchHistoricTrades } from '../../api/Historic';
import Toast from '../Toast/Toastpopup';

const Historic = () => {
  const { mode } = useTradingMode();
  const [trades, setTrades] = useState([]);
  const [toast, setToast] = useState(null);
  const [loading, setLoading] = useState(true);
  const [fadeIn, setFadeIn] = useState(false);
  const [grouped, setGrouped] = useState({ positive: {}, negative: {} });

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      const result = await fetchHistoricTrades(mode);

      if (result.success) {
        setTrades(result.trades);
      } else {
        setTrades([]);
        setToast({
          message: result.error || 'Failed to fetch trade data.',
          type: 'error',
        });
      }

      setLoading(false);
    };

    fetchData();
  }, [mode]);

  // Group trades into histogram buckets
  useEffect(() => {
    const pos = {};
    const neg = {};

    trades.forEach((t) => {
      const pnl = parseFloat(t.pnl_percent || 0);
      const bucket = Math.floor(Math.abs(pnl) / 10) * 10;
      if (pnl >= 0) {
        pos[bucket] = (pos[bucket] || 0) + 1;
      } else {
        neg[bucket] = (neg[bucket] || 0) + 1;
      }
    });

    setGrouped({ positive: pos, negative: neg });
  }, [trades]);

  const maxCount = Math.max(
    ...Object.values(grouped.positive),
    ...Object.values(grouped.negative),
    1,
  );

  useEffect(() => {
    if (!loading) {
      const timeout = setTimeout(() => setFadeIn(true), 100);
      return () => clearTimeout(timeout);
    }
  }, [loading]);

  const totalPnl = trades.reduce((acc, t) => acc + parseFloat(t.pnl_absolute || 0), 0);
  const pnlColor = totalPnl >= 0 ? styles.greenText : styles.redText;

  return (
    <>
      <div className={`${styles.container} ${fadeIn ? styles.fadeIn : ''}`}>
        <h3 className={styles.title}>
          {mode === 'paper' ? 'Paper Trades' : 'Live Trades'}
        </h3>

        <div className={styles.histogramWrapper}>
          <div className={styles.axisLabel}>PROFIT</div>
          {Object.keys(grouped.positive)
            .sort((a, b) => b - a)
            .map((bucket) => {
              const count = grouped.positive[bucket];
              const width = Math.round((count / maxCount) * 180);
              return (
                <div key={`p${bucket}`} className={styles.barRow}>
                  <span className={styles.barLabel}>{bucket}%</span>
                  <div
                    className={`${styles.bar} ${styles.profit}`}
                    style={{ width: `${width}px` }}
                  >
                    {count}
                  </div>
                </div>
              );
            })}

          <div className={styles.axisLabel}>LOSS</div>
          {Object.keys(grouped.negative)
            .sort((a, b) => a - b)
            .map((bucket) => {
              const count = grouped.negative[bucket];
              const width = Math.round((count / maxCount) * 180);
              return (
                <div key={`n${bucket}`} className={styles.barRow}>
                  <span className={styles.barLabel}>{bucket}%</span>
                  <div
                    className={`${styles.bar} ${styles.loss}`}
                    style={{ width: `${width}px` }}
                  >
                    {count}
                  </div>
                </div>
              );
            })}
        </div>

        <div className={styles.summaryBox}>
          <p>
            Total of last {trades.length} trades:{' '}
            <span className={pnlColor}>${totalPnl.toFixed(2)}</span>
          </p>
        </div>
      </div>
      {toast && (
        <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />
      )}
    </>
  );
};

export default Historic;
