import PropTypes from 'prop-types';
import styles from './styles/TooltipCandle.module.css'; // We can reuse the same style

const TooltipEvent = ({ eventData }) => {
  if (!eventData) return null;

  // Helper to format values, showing 'N/A' for null/undefined
  const formatValue = (value) => value ?? 'N/A';

  return (
    <div className={styles.tooltip_event}>
      {' '}
      {/* Use a new class or reuse */}
      <div className={styles.text}>
        <strong>{eventData.event}</strong>
        <hr className={styles.hr} />
        <span>
          <strong>Time:</strong> {eventData.time} ({eventData.currency})
        </span>
        <span>
          <strong>Impact:</strong> {eventData.impact}
        </span>
        <hr className={styles.hr} />
        <span>
          <strong>Actual:</strong> {formatValue(eventData.actual)}
        </span>
        <span>
          <strong>Forecast:</strong> {formatValue(eventData.forecast)}
        </span>
        <span>
          <strong>Previous:</strong> {formatValue(eventData.previous)}
        </span>
      </div>
    </div>
  );
};

TooltipEvent.propTypes = {
  eventData: PropTypes.shape({
    event: PropTypes.string,
    time: PropTypes.string,
    currency: PropTypes.string,
    impact: PropTypes.string,
    forecast: PropTypes.number,
    previous: PropTypes.number,
  }).isRequired,
};

export default TooltipEvent;
