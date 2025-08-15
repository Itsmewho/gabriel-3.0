import { useMemo, useCallback } from 'react';
import PropTypes from 'prop-types';
import styles from './styles/PriceBar.module.css';

const PriceBar = ({ marketData = [], currentPrice = 0, selectedHours }) => {
  const lastHoursData = useMemo(() => {
    if (!marketData.length) return [];
    const lastCandleTime = new Date(marketData[marketData.length - 1].time);
    const cutoffTime = new Date(
      lastCandleTime.getTime() - selectedHours * 60 * 60 * 1000,
    );

    return marketData.filter((entry) => new Date(entry.time) >= cutoffTime);
  }, [marketData, selectedHours]);

  const highPrice = useMemo(() => {
    if (!lastHoursData.length) return 0;
    return Math.max(...lastHoursData.map((entry) => entry.high));
  }, [lastHoursData]);

  const lowPrice = useMemo(() => {
    if (!lastHoursData.length) return 0;
    return Math.min(...lastHoursData.map((entry) => entry.low));
  }, [lastHoursData]);

  const priceSteps = useMemo(() => {
    if (highPrice === 0 || lowPrice === 0) return [];
    const stepSize = (highPrice - lowPrice) / 10;
    return [...Array(10)].map((_, i) => (lowPrice + i * stepSize).toFixed(5)).reverse();
  }, [highPrice, lowPrice]);

  const getPricePosition = useCallback(
    (price) => {
      if (!highPrice || !lowPrice || price == null) return '50%';
      return `${((highPrice - price) / (highPrice - lowPrice)) * 100}%`;
    },
    [highPrice, lowPrice],
  );

  return (
    <div className={styles.price_bar_container}>
      <div className={styles.price_labels}>
        <div className={styles.high_label} style={{ top: getPricePosition(highPrice) }}>
          {highPrice.toFixed(5)}
        </div>

        {priceSteps.map((price, index) => (
          <div key={index} className={styles.price_step}>
            {price}
          </div>
        ))}

        <div
          className={styles.current_label}
          style={{ top: getPricePosition(currentPrice) }}
        >
          {currentPrice.toFixed(5)}
        </div>

        <div className={styles.low_label} style={{ top: getPricePosition(lowPrice) }}>
          {lowPrice.toFixed(5)}
        </div>
      </div>
    </div>
  );
};

PriceBar.propTypes = {
  marketData: PropTypes.arrayOf(
    PropTypes.shape({
      time: PropTypes.string.isRequired,
      high: PropTypes.number.isRequired,
      low: PropTypes.number.isRequired,
    }),
  ).isRequired,
  currentPrice: PropTypes.number.isRequired,
  selectedHours: PropTypes.number.isRequired,
};

export default PriceBar;
