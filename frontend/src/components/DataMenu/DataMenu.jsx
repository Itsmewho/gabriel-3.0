import PropTypes from 'prop-types';
import styles from './styles/Data.module.css';
import Tooltip from '../Tooltip/Tooltip';
import { useToolContext } from '../../context/ToolContext';

const DataMenu = ({ onSelectView }) => {
  const { isCrosshairActive, setIsCrosshairActive } = useToolContext();

  return (
    <div className={styles.dataContainer}>
      <div className={styles.iconContainer}>
        <Tooltip tooltipText="Data" position="left" onClick={() => onSelectView('data')}>
          <div className={styles.iconData}></div>
        </Tooltip>
        <Tooltip tooltipText="Cursor" position="left">
          <div
            role="button"
            tabIndex={0}
            aria-pressed={isCrosshairActive}
            className={`${styles.Crosshair} ${isCrosshairActive ? 'active' : ''}`}
            onClick={() => setIsCrosshairActive(!isCrosshairActive)}
            onKeyDown={(e) => {
              if (e.key === '`' || e.key === ' ') {
                setIsCrosshairActive((prev) => !prev);
                e.preventDefault(); // optional for 'Space'
              }
            }}
          ></div>
        </Tooltip>
      </div>
      <div className={styles.dataContainer}>
        <Tooltip
          tooltipText="Calendar"
          position="left"
          onClick={() => onSelectView('calander')}
        >
          <div className={styles.iconCalander}></div>
        </Tooltip>
        <Tooltip
          tooltipText="Order"
          position="left"
          onClick={() => onSelectView('order')}
        >
          <div className={styles.iconOrder}></div>
        </Tooltip>
        <Tooltip
          tooltipText="Open Orders"
          position="left"
          onClick={() => onSelectView('openorder')}
        >
          <div className={styles.iconOpen}></div>
        </Tooltip>
        <Tooltip
          tooltipText="Historic"
          position="left"
          onClick={() => onSelectView('historic')}
        >
          <div className={styles.iconHistoric}></div>
        </Tooltip>
        <Tooltip
          tooltipText="Notifications"
          position="left"
          onClick={() => onSelectView('notifications')}
        >
          <div className={styles.iconNotifications}></div>
        </Tooltip>
      </div>
    </div>
  );
};

DataMenu.propTypes = {
  onSelectView: PropTypes.func.isRequired,
  forecastView: PropTypes.func.isRequired,
  onPlayBall: PropTypes.func.isRequired,
};

export default DataMenu;
