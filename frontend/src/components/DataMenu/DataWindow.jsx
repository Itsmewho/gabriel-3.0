import PropTypes from 'prop-types';
import styles from './styles/Data.module.css';
import Data from './Data';
import Historic from './Historic';
import Calander from './Calander';
import OpenOrder from './OpenOrder';
import Notifications from './Notifications';
import Order from './Order';

const DataWindow = ({ selectedView }) => {
  const renderView = () => {
    switch (selectedView || 'data') {
      case 'data':
        return <Data />;
      case 'historic':
        return <Historic />;
      case 'calander':
        return <Calander />;
      case 'order':
        return <Order />;
      case 'openorder':
        return <OpenOrder />;
      case 'notifications':
        return <Notifications />;
      default:
        return <Data />;
    }
  };

  return <div className={`${styles.dataWindow} ${styles.fadeIn}`}>{renderView()}</div>;
};

DataWindow.propTypes = {
  selectedView: PropTypes.oneOf([
    'data',
    'historic',
    'calander',
    'order',
    'notifications',
    'openorder',
  ]).isRequired,
};

export default DataWindow;
