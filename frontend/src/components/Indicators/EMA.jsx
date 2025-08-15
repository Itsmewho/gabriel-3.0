import { useMemo } from 'react';
import PropTypes from 'prop-types';
import styles from './styles/EMA.module.css';
import { useApi } from '../../context/ApiContext';

const EMAChart = ({ marketData, highPrice, lowPrice, timelineKeys }) => {
  const { emaData } = useApi();

  const aligned = useMemo(() => {
    if (!Array.isArray(emaData) || !emaData.length || !Array.isArray(timelineKeys))
      return [];
    const map = new Map(emaData.map((p) => [String(p.time).slice(0, 16), p.value]));
    return timelineKeys.map((k) => (map.has(k) ? { time: k, value: map.get(k) } : null));
  }, [emaData, timelineKeys]);

  return (
    <div className={styles.emaContainer}>
      <div className={styles.emaChart}>
        {aligned.map((pt, idx) => {
          if (!pt || highPrice === lowPrice)
            return <div key={idx} className={styles.emaPointEmpty} />;
          const relY = ((pt.value - lowPrice) / (highPrice - lowPrice)) * 100;
          return (
            <div
              key={idx}
              className={styles.emaPoint}
              style={{ bottom: `${relY}%`, left: `${(idx / aligned.length) * 100}%` }}
            />
          );
        })}
      </div>
    </div>
  );
};

EMAChart.propTypes = {
  marketData: PropTypes.arrayOf(PropTypes.object).isRequired,
  highPrice: PropTypes.number.isRequired,
  lowPrice: PropTypes.number.isRequired,
};

export default EMAChart;
