import PropTypes from 'prop-types';
import { useMemo } from 'react';
import { useApi } from '../../context/ApiContext';
import styles from './styles/BB.module.css';

const BollingerBands = ({ marketData }) => {
  const { bollingerData: bbData } = useApi();

  const alignedBbData = useMemo(() => {
    if (!Array.isArray(bbData) || !bbData.length || !marketData.length) return [];

    const reversedBbData = [...bbData].reverse();
    const bbDataMap = new Map(
      reversedBbData.map((item) => [new Date(item.time).getTime(), item]),
    );

    return marketData.map((candle) => {
      const candleTime = new Date(candle.time).getTime();
      return bbDataMap.get(candleTime) || null;
    });
  }, [bbData, marketData]);

  const { bandHigh, bandLow } = useMemo(() => {
    if (!alignedBbData.length) return { bandHigh: 0, bandLow: 0 };
    const validPoints = alignedBbData.filter((p) => p);
    if (!validPoints.length) return { bandHigh: 0, bandLow: 0 };

    const high = Math.max(...validPoints.map((p) => p.upper));
    const low = Math.min(...validPoints.map((p) => p.lower));
    return { bandHigh: high, bandLow: low };
  }, [alignedBbData]);

  const getBandStyle = (topPrice, bottomPrice) => {
    if (bandHigh === bandLow) return { display: 'none' };
    const heightPercent = ((topPrice - bottomPrice) / (bandHigh - bandLow)) * 100;
    const bottomPercent = ((bottomPrice - bandLow) / (bandHigh - bandLow)) * 100;
    return {
      height: `${heightPercent}%`,
      bottom: `${bottomPercent}%`,
    };
  };

  const getLineStyle = (price) => {
    if (bandHigh === bandLow) return { display: 'none' };
    const bottomPercent = ((price - bandLow) / (bandHigh - bandLow)) * 100;
    return {
      bottom: `${bottomPercent}%`,
    };
  };

  return (
    <div className={styles.subChartContainer}>
      {alignedBbData.map((bbPoint, index) => {
        if (!bbPoint) {
          return <div key={index} className={styles.bbStep} />;
        }
        return (
          <div key={new Date(bbPoint.time).getTime()} className={styles.bbStep}>
            <div
              className={styles.bbFill}
              style={getBandStyle(bbPoint.upper, bbPoint.lower)}
            />
            <div className={styles.bbMavg} style={getLineStyle(bbPoint.middle)} />
          </div>
        );
      })}
    </div>
  );
};

BollingerBands.propTypes = {
  marketData: PropTypes.arrayOf(PropTypes.object).isRequired,
};

export default BollingerBands;
