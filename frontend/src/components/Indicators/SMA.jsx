import { useMemo } from 'react';
import PropTypes from 'prop-types';
import styles from './styles/SMA.module.css';
import { useApi } from '../../context/ApiContext';

const SMAChart = ({ marketData, highPrice, lowPrice, timelineKeys }) => {
  const { smaData } = useApi();

  const aligned = useMemo(() => {
    if (!Array.isArray(smaData) || !smaData.length || !Array.isArray(timelineKeys))
      return [];
    const map = new Map(smaData.map((p) => [String(p.time).slice(0, 16), p.value]));
    return timelineKeys.map((k) => (map.has(k) ? { time: k, value: map.get(k) } : null));
  }, [smaData, timelineKeys]);

  return (
    <div className={styles.smaContainer}>
      <div className={styles.smaChart}>
        {aligned.map((pt, idx) => {
          if (!pt || highPrice === lowPrice)
            return <div key={idx} className={styles.smaPointEmpty} />;
          const relY = ((pt.value - lowPrice) / (highPrice - lowPrice)) * 100;
          return (
            <div
              key={idx}
              className={styles.smaPoint}
              style={{ bottom: `${relY}%`, left: `${(idx / aligned.length) * 100}%` }}
            />
          );
        })}
      </div>
    </div>
  );
};

SMAChart.propTypes = {
  marketData: PropTypes.arrayOf(PropTypes.object).isRequired,
  highPrice: PropTypes.number.isRequired,
  lowPrice: PropTypes.number.isRequired,
};

export default SMAChart;
