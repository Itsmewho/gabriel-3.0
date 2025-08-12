import { useMemo } from 'react';
import PropTypes from 'prop-types';
import styles from './styles/SMA.module.css';
import { useApi } from '../../context/ApiContext';

const SMAChart = ({ marketData, highPrice, lowPrice }) => {
  const { smaData } = useApi();

  const alignedSmaData = useMemo(() => {
    if (!Array.isArray(smaData) || !smaData.length || !marketData.length) {
      return [];
    }

    // The API sends data newest-first, but the chart displays oldest-first.
    const reversedSmaData = [...smaData].reverse();

    // Create a Map for fast lookups of SMA data by its timestamp.
    const smaDataMap = new Map(
      reversedSmaData.map((item) => [new Date(item.time).getTime(), item]),
    );

    // Iterate over the main marketData array. For each candle, find the
    // corresponding SMA data. If it doesn't exist, use null.
    return marketData.map((candle) => {
      const candleTime = new Date(candle.time).getTime();
      return smaDataMap.get(candleTime) || null;
    });
  }, [smaData, marketData]);

  return (
    <div className={styles.smaContainer}>
      {marketData.length > 0 ? (
        <div className={styles.smaChart}>
          {alignedSmaData.map((sma, idx) => {
            if (!sma) {
              return <div key={idx} className={styles.smaPointEmpty} />;
            }

            const relY = ((sma.value - lowPrice) / (highPrice - lowPrice)) * 100;
            return (
              <div
                key={idx}
                className={styles.smaPoint}
                style={{
                  bottom: `${relY}%`,
                  left: `${(idx / marketData.length) * 100}%`,
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

SMAChart.propTypes = {
  marketData: PropTypes.arrayOf(PropTypes.object).isRequired,
  highPrice: PropTypes.number.isRequired,
  lowPrice: PropTypes.number.isRequired,
};

export default SMAChart;
