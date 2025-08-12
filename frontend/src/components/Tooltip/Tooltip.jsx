import PropTypes from 'prop-types';
import styles from '../Tooltip/styles/Tooltip.module.css';
import clsx from 'clsx';

const Tooltip = ({ children, tooltipText, position = 'top', onClick }) => {
  return (
    // eslint-disable-next-line jsx-a11y/click-events-have-key-events, jsx-a11y/no-static-element-interactions
    <div
      className={clsx(styles.tooltip_component, styles[`tooltip-${position}_component`])}
      onClick={onClick}
    >
      {children}
      <span className={styles.tooltiptext_component}>{tooltipText}</span>
    </div>
  );
};

Tooltip.propTypes = {
  children: PropTypes.node.isRequired,
  tooltipText: PropTypes.string.isRequired,
  position: PropTypes.oneOf(['top', 'right', 'bottom', 'left']),
  onClick: PropTypes.func,
};

export default Tooltip;
