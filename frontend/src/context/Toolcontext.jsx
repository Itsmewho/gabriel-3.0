import { createContext, useContext, useState } from 'react';
import PropTypes from 'prop-types';

const ToolContext = createContext();

export const ToolProvider = ({ children }) => {
  const [isCrosshairActive, setIsCrosshairActive] = useState(false);

  return (
    <ToolContext.Provider value={{ isCrosshairActive, setIsCrosshairActive }}>
      {children}
    </ToolContext.Provider>
  );
};

ToolProvider.propTypes = {
  children: PropTypes.node.isRequired,
};

export const useToolContext = () => {
  const context = useContext(ToolContext);
  if (!context) {
    throw new Error('useToolContext must be used within a ToolProvider');
  }
  return context;
};
