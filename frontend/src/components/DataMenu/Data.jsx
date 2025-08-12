import { useTradingMode } from '../../context/TradingmodeContext';
import { useApi } from '../../context/ApiContext';
import styles from './styles/data.module.css';

const Data = () => {
  const { mode } = useTradingMode();
  const selectedSymbol = 'EURUSD';
  const selectedTimeframe = '1m';

  const { trendMarketData, markovData } = useApi();
  const latestCandle = trendMarketData.at(-1);
  const isLoading = !latestCandle;

  return (
    <div className={styles.view}>
      <div className={styles.toggleWrapper}>
        <span className={styles.modeLabel}>
          {mode === 'live' ? 'Live Mode' : 'Paper Mode'}
        </span>
      </div>

      {isLoading ? (
        <p>Loading...</p>
      ) : (
        <div className={styles.dataBox}>
          <div className={styles.spacing}>
            <p>Symbol:</p>
            <p>{selectedSymbol}</p>
          </div>
          <div className={styles.spacing}>
            <p>Timeframe:</p>
            <p>{selectedTimeframe}</p>
          </div>
          <div className={styles.spacing}>
            <p>Time:</p>
            <p>{new Date(latestCandle.time).toISOString().slice(11, 16)}</p>
          </div>
          <div className={styles.spacing}>
            <p>Open:</p>
            <p>{latestCandle.open.toFixed(5)}</p>
          </div>
          <div className={styles.spacing}>
            <p>Current:</p>
            <p>{latestCandle.close.toFixed(5)}</p>
          </div>
          <div className={styles.spacing}>
            <p>High:</p>
            <p className="green">{latestCandle.high.toFixed(5)}</p>
          </div>
          <div className={styles.spacing}>
            <p>Low:</p>
            <p className="red">{latestCandle.low.toFixed(5)}</p>
          </div>
          <div className={styles.spacing}>
            <p>Change:</p>
            <p>{parseFloat(latestCandle.change).toFixed(5)}</p>
          </div>
          <div className={styles.spacing}>
            <p>Change %:</p>
            <p>{parseFloat(latestCandle.change_percent).toFixed(2)}%</p>
          </div>
          <div className={styles.spacing}>
            <p>Volume:</p>
            <p className="green">{latestCandle.tick_volume}</p>
          </div>

          {markovData && (
            <>
              <hr className={styles.divider} />
              <div className={styles.spacing}>
                <h4>Markov Prediction</h4>
              </div>
              <div className={styles.spacing}>
                <p>Current:</p>
                <p className={styles[markovData.current_state.toLowerCase()]}>
                  {markovData.current_state}
                </p>
              </div>
              <div className={styles.spacing}>
                <p>Next:</p>
                <p className={styles[markovData.predicted_next_state.toLowerCase()]}>
                  {markovData.predicted_next_state}
                </p>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default Data;
