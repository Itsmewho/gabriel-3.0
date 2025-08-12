import { createContext, useState, useCallback, useContext } from 'react';
import Toast from '../components/Toast/Toastpopup';

const ToastContext = createContext();

export const useToast = () => useContext(ToastContext);

export const ToastProvider = ({ children }) => {
  const [toast, setToast] = useState(null);

  const showToast = useCallback((message, type = 'info') => {
    setToast({ message, type });
  }, []);

  return (
    <ToastContext.Provider value={{ showToast }}>
      {children}
      {toast && (
        <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />
      )}
    </ToastContext.Provider>
  );
};

import PropTypes from 'prop-types';

ToastProvider.propTypes = {
  children: PropTypes.node.isRequired,
};
