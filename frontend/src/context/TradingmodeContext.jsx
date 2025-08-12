import { createContext, useContext, useState } from 'react';
import PropTypes from 'prop-types';
import Toast from '../components/Toast/Toastpopup';
import { setTradingMode } from '../api/modeApi';

const MODES = {
  LIVE: 'live',
  PAPER: 'paper',
};

const TradingModeContext = createContext();

export const TradingModeProvider = ({ children }) => {
  const [mode, setMode] = useState(MODES.LIVE);
  const [toast, setToast] = useState(null);

  const showToast = (message, type = 'info') => {
    setToast({ message, type });
  };

  const setModeAndNotifyBackend = async (newModeOrUpdater) => {
    const newMode =
      typeof newModeOrUpdater === 'function' ? newModeOrUpdater(mode) : newModeOrUpdater;

    if (!Object.values(MODES).includes(newMode)) {
      showToast(`Invalid mode: ${newMode}`, 'error');
      return;
    }

    setMode(newMode);

    try {
      await setTradingMode(newMode);
      showToast(`Mode set to "${newMode}"`, 'success');
    } catch (error) {
      showToast(`Failed to set mode: ${error.message}`, 'error');
    }
  };

  return (
    <TradingModeContext.Provider value={{ mode, setMode: setModeAndNotifyBackend }}>
      <>
        {children}
        {toast && (
          <Toast
            message={toast.message}
            type={toast.type}
            onClose={() => setToast(null)}
          />
        )}
      </>
    </TradingModeContext.Provider>
  );
};

TradingModeProvider.propTypes = {
  children: PropTypes.node.isRequired,
};

export const useTradingMode = () => {
  const context = useContext(TradingModeContext);
  if (!context) {
    throw new Error('useTradingMode must be used within a TradingModeProvider');
  }
  return context;
};

export const TRADING_MODES = MODES;
