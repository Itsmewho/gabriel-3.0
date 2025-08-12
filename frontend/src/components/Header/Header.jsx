/* eslint-disable jsx-a11y/label-has-associated-control */
import { useState } from 'react';
import styles from './styles/Header.module.css';
import Tooltip from '../Tooltip/Tooltip';
import Toast from '../Toast/Toastpopup';
import { useTradingMode } from '../../context/TradingmodeContext';

const Header = () => {
  const { mode, setMode } = useTradingMode();

  const [toast, setToast] = useState(null);

  return (
    <>
      <header className={styles.header}>
        <div className={styles.navContainer}>
          <div className={styles.modeContainer}>
            <Tooltip tooltipText="switch app mode" position="bottom">
              <div className={styles.modeToggleWrapper}>
                <span className={styles.modeLabel}>
                  {mode === 'live' ? 'Live' : 'Paper'}
                </span>
                <label className={styles.modeSwitch}>
                  <input
                    type="checkbox"
                    checked={mode === 'paper'}
                    onChange={() =>
                      setMode((prev) => (prev === 'live' ? 'paper' : 'live'))
                    }
                  />
                  <span className={styles.modeSlider} />
                </label>
              </div>
            </Tooltip>
          </div>
        </div>
      </header>
      {toast && (
        <Toast type={toast.type} message={toast.message} onClose={() => setToast(null)} />
      )}
    </>
  );
};

export default Header;
