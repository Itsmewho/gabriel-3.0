import { useState } from 'react';
import PropTypes from 'prop-types';
import styles from './styles/Event.module.css';
import TooltipEvent from '../Tooltip/TooltipEvents';

const getImpactClass = (impactString) => {
  if (typeof impactString !== 'string') {
    return styles.lowImpact; // Default for safety
  }

  const lowercasedImpact = impactString.toLowerCase();

  if (lowercasedImpact.includes('high')) {
    return styles.highImpact;
  }
  if (lowercasedImpact.includes('medium')) {
    return styles.mediumImpact;
  }
  if (lowercasedImpact.includes('Non')) {
    return styles.NonImpact;
  }

  return styles.lowImpact;
};

const EventMarker = ({ event }) => {
  const [isHovered, setIsHovered] = useState(false);

  const impactClass = getImpactClass(event.impact);

  return (
    <div
      className={styles.eventMarkerContainer}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className={`${styles.eventMarker} ${impactClass}`} />
      {isHovered && <TooltipEvent eventData={event} />}
    </div>
  );
};

EventMarker.displayName = 'EventMarker';

EventMarker.propTypes = {
  event: PropTypes.shape({
    id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    impact: PropTypes.string,
    event: PropTypes.string,
    time: PropTypes.string,
    currency: PropTypes.string,
    actual: PropTypes.number,
    forecast: PropTypes.number,
    previous: PropTypes.number,
  }).isRequired,
};

export default EventMarker;
