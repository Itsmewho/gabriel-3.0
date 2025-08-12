import { useEffect } from 'react';
import PropTypes from 'prop-types';
import styles from './styles/Toast.module.css';

const Toast = ({ message, type = 'info', onClose }) => {
  useEffect(() => {
    const timer = setTimeout(onClose, 2000);
    return () => clearTimeout(timer);
  }, [onClose]);

  return <div className={`${styles.toast} ${styles[type]}`}>{message}</div>;
};

Toast.propTypes = {
  message: PropTypes.string.isRequired,
  type: PropTypes.oneOf(['info', 'success', 'error']),
  onClose: PropTypes.func.isRequired,
};

export default Toast;
