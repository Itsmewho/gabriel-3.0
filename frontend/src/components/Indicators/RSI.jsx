import { useMemo } from 'react';
import PropTypes from 'prop-types';
import styles from './styles/RSI.module.css';
import { useApi } from '../../context/ApiContext';

const RSIChart = ({ marketData }) => {
  const { rsiData } = useApi();

  const alignedRsiData = useMemo(() => {
    // Return an empty array if the necessary data isn't available
    if (!Array.isArray(rsiData) || !rsiData.length || !marketData.length) {
      return [];
    }

    // ✅ FIX 1: Reverse the indicator data to match the chart's ascending time order.
    // The API sends data newest-first, but the chart displays oldest-first.
    const reversedRsiData = [...rsiData].reverse();

    // Create a Map for fast lookups of RSI data by its timestamp.
    const rsiDataMap = new Map(
      reversedRsiData.map((item) => [new Date(item.time).getTime(), item]),
    );

    // ✅ FIX 2: Iterate over the main marketData array to guarantee order and length.
    // This ensures the indicator chart has the same number of data points as the main chart.
    return marketData.map((candle) => {
      const candleTime = new Date(candle.time).getTime();
      // If a matching RSI point is found, use it; otherwise, use null to create a gap.
      return rsiDataMap.get(candleTime) || null;
    });
  }, [rsiData, marketData]);

  return (
    <div className={styles.rsiContainer}>
      <div className={styles.rsiChart}>
        {alignedRsiData.map((rsi, idx) => {
          // If the RSI data for this candle is null, render an empty placeholder
          // to maintain alignment and prevent gaps.
          if (!rsi) {
            return (
              <div key={idx} className={styles.rsiBar} style={{ visibility: 'hidden' }} />
            );
          }

          return (
            <div
              key={idx}
              className={styles.rsiBar}
              style={{
                height: `${(rsi.value / 100) * 100}%`,
                backgroundColor:
                  rsi.value > 70 ? '#f23746' : rsi.value < 30 ? '#08ab70' : '#0044aa',
              }}
            />
          );
        })}
      </div>
    </div>
  );
};

RSIChart.propTypes = {
  marketData: PropTypes.arrayOf(PropTypes.object).isRequired,
};

export default RSIChart;
