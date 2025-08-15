import { useMemo } from 'react';
import PropTypes from 'prop-types';
import styles from './styles/RSI.module.css';
import { useApi } from '../../context/ApiContext';

const RSIChart = ({ marketData, timelineKeys }) => {
  const { rsiData } = useApi();

  const aligned = useMemo(() => {
    if (!Array.isArray(rsiData) || !rsiData.length || !Array.isArray(timelineKeys))
      return [];
    const map = new Map(rsiData.map((p) => [String(p.time).slice(0, 16), p.value]));
    return timelineKeys.map((k) => (map.has(k) ? { time: k, value: map.get(k) } : null));
  }, [rsiData, timelineKeys]);

  return (
    <div className={styles.rsiContainer}>
      <div className={styles.rsiChart}>
        {aligned.map((pt, idx) =>
          pt ? (
            <div
              key={idx}
              className={styles.rsiBar}
              style={{
                height: `${pt.value}%`,
                backgroundColor:
                  pt.value > 70 ? '#f23746' : pt.value < 30 ? '#08ab70' : '#0044aa',
              }}
            />
          ) : (
            <div key={idx} className={styles.rsiBar} style={{ visibility: 'hidden' }} />
          ),
        )}
      </div>
    </div>
  );
};

RSIChart.propTypes = {
  marketData: PropTypes.arrayOf(PropTypes.object).isRequired,
};

export default RSIChart;
