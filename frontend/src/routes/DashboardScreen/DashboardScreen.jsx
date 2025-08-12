import { useState } from 'react';

import TrendWindow from '../../components/Trend/TrendWindow';
import DataWindow from '../../components/DataMenu/dataWindow';
import DataMenu from '../../components/DataMenu/DataMenu';
import styles from './styles/DashBoard.module.css';

function DashboardScreen() {
  const [selectedView, setSelectedView] = useState('data');
  return (
    <div className={styles.dashboardContainer}>
      <div className={styles.dashboardGrid}>
        <div className={styles.trendWindow}>
          <TrendWindow />
        </div>
        <div className={styles.dataWindow}>
          <DataWindow selectedView={selectedView} />
        </div>
        <div className={styles.sideMenu}>
          <DataMenu onSelectView={setSelectedView} />
        </div>
        <div className={styles.modelWindow}>mertics</div>
      </div>
    </div>
  );
}

export default DashboardScreen;
