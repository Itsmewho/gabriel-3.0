import { useMemo } from 'react';
import PropTypes from 'prop-types';
import styles from './styles/EMA.module.css';
import { useApi } from '../../context/ApiContext';

const EMAChart = ({ marketData, highPrice, lowPrice }) => {
  const { emaData } = useApi();

  const alignedEmaData = useMemo(() => {
    if (!Array.isArray(emaData) || !emaData.length || !marketData.length) return [];

    const marketTimestamps = new Set(
      marketData.map((entry) => entry.time.substring(0, 19)),
    );

    return emaData.filter((ema) => {
      const emaTimestamp = ema.time.substring(0, 19);
      return marketTimestamps.has(emaTimestamp);
    });
  }, [emaData, marketData]);

  return (
    <div className={styles.emaContainer}>
      {alignedEmaData.length > 0 ? (
        <div className={styles.emaChart}>
          {alignedEmaData.map((ema, idx) => {
            const relY = ((ema.value - lowPrice) / (highPrice - lowPrice)) * 100;
            return (
              <div
                key={idx}
                className={styles.emaPoint}
                style={{
                  bottom: `${relY}%`,
                  left: `${(idx / alignedEmaData.length) * 100}%`,
                }}
              />
            );
          })}
        </div>
      ) : (
        <p className={styles.noData}>Loading</p>
      )}
    </div>
  );
};

EMAChart.propTypes = {
  marketData: PropTypes.arrayOf(PropTypes.object).isRequired,
  highPrice: PropTypes.number.isRequired,
  lowPrice: PropTypes.number.isRequired,
};

export default EMAChart;
