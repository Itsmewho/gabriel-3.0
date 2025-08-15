import { useMemo } from 'react';
import PropTypes from 'prop-types';
import styles from './styles/Markov.module.css';
import { useApi } from '../../context/ApiContext';

const MarkovChart = ({ marketData, timelineKeys }) => {
  const { markovHistory } = useApi();

  const aligned = useMemo(() => {
    if (
      !Array.isArray(markovHistory) ||
      !markovHistory.length ||
      !Array.isArray(timelineKeys)
    )
      return [];
    const map = new Map(markovHistory.map((p) => [String(p.time).slice(0, 16), p.state]));
    return timelineKeys.map((k) => (map.has(k) ? { time: k, state: map.get(k) } : null));
  }, [markovHistory, timelineKeys]);

  return (
    <div className={styles.markovContainer}>
      <div className={styles.markovChart}>
        {aligned.map((pt, idx) => (
          <div
            key={idx}
            className={`${styles.markovBar} ${
              pt?.state === 'Bullish'
                ? styles.bullish
                : pt?.state === 'Bearish'
                ? styles.bearish
                : styles.neutral
            }`}
            style={{
              height:
                pt?.state === 'Bullish' ? '50%' : pt?.state === 'Bearish' ? '40%' : '15%',
              left: `${(idx / aligned.length) * 100}%`,
            }}
          />
        ))}
      </div>
    </div>
  );
};

MarkovChart.propTypes = {
  marketData: PropTypes.arrayOf(PropTypes.object).isRequired,
};

export default MarkovChart;
