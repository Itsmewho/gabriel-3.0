import PropTypes from 'prop-types';
import styles from './styles/TooltipCandle.module.css';
import { useMemo } from 'react';

const TooltipCandle = ({ x, y, candleData, containerRef }) => {
  const { adjustedX, adjustedY } = useMemo(() => {
    if (!containerRef?.current) {
      return { adjustedX: x + 10, adjustedY: y - 30 };
    }

    const { width, height } = containerRef.current.getBoundingClientRect();
    const tooltipWidth = 140;
    const tooltipHeight = 90;
    const padding = 15;

    let adjX = x + 10;
    let adjY = y - 30;

    if (x + tooltipWidth + padding > width) {
      adjX = x - tooltipWidth - 10;
    }
    if (y + tooltipHeight + padding > height) {
      adjY = y - tooltipHeight - 10;
    }

    return { adjustedX: adjX, adjustedY: adjY };
  }, [x, y, containerRef]);

  if (!candleData) return null;

  return (
    <div
      className={styles.tooltip}
      style={{ left: `${adjustedX}px`, top: `${adjustedY}px` }}
    >
      <span className={styles.text}>
        <strong>Time:</strong> {new Date(candleData.time).toISOString().slice(11, 16)}
        <br />
        <strong>Open:</strong> {candleData.open.toFixed(5)}
        <br />
        <strong>High:</strong> {candleData.high.toFixed(5)}
        <br />
        <strong>Low:</strong> {candleData.low.toFixed(5)}
        <br />
        <strong>Close:</strong> {candleData.close.toFixed(5)}
        <br />
        <strong>Change:</strong> {parseFloat(candleData.change).toFixed(5)}
        <br />
        <strong>Change %:</strong> {parseFloat(candleData.change_percent).toFixed(2)}%
      </span>
    </div>
  );
};

TooltipCandle.propTypes = {
  x: PropTypes.number.isRequired,
  y: PropTypes.number.isRequired,
  candleData: PropTypes.shape({
    time: PropTypes.string.isRequired,
    open: PropTypes.number.isRequired,
    high: PropTypes.number.isRequired,
    low: PropTypes.number.isRequired,
    close: PropTypes.number.isRequired,
    change: PropTypes.string.isRequired,
    change_percent: PropTypes.number.isRequired,
  }),
  containerRef: PropTypes.shape({ current: PropTypes.instanceOf(Element) }),
};

export default TooltipCandle;
