import PropTypes from 'prop-types';
import { useMemo } from 'react';
import { useApi } from '../../context/ApiContext';
import styles from './styles/BB.module.css';

const BollingerBands = ({ marketData, timelineKeys }) => {
  const { bollingerData: bbData } = useApi();

  const aligned = useMemo(() => {
    if (!Array.isArray(bbData) || !bbData.length || !Array.isArray(timelineKeys))
      return [];
    const map = new Map(
      bbData.map((p) => [new Date(p.time).toISOString().slice(0, 16), p]),
    );
    return timelineKeys.map((k) => map.get(k) || null);
  }, [bbData, timelineKeys]);

  const valid = aligned.filter(Boolean);
  const bandHigh = valid.length ? Math.max(...valid.map((p) => p.upper)) : 0;
  const bandLow = valid.length ? Math.min(...valid.map((p) => p.lower)) : 0;

  const getBandStyle = (top, bottom) => {
    if (bandHigh === bandLow) return { display: 'none' };
    return {
      height: `${((top - bottom) / (bandHigh - bandLow)) * 100}%`,
      bottom: `${((bottom - bandLow) / (bandHigh - bandLow)) * 100}%`,
    };
  };
  const getLineStyle = (price) => {
    if (bandHigh === bandLow) return { display: 'none' };
    return { bottom: `${((price - bandLow) / (bandHigh - bandLow)) * 100}%` };
  };

  return (
    <div className={styles.subChartContainer}>
      {aligned.map((bb, i) =>
        bb ? (
          <div key={i} className={styles.bbStep}>
            <div className={styles.bbFill} style={getBandStyle(bb.upper, bb.lower)} />
            <div className={styles.bbMavg} style={getLineStyle(bb.middle)} />
          </div>
        ) : (
          <div key={i} className={styles.bbStep} />
        ),
      )}
    </div>
  );
};

BollingerBands.propTypes = {
  marketData: PropTypes.arrayOf(PropTypes.object).isRequired,
};

export default BollingerBands;
