import { StrictMode } from 'react';
import ReactDOM from 'react-dom/client';
import {
  createBrowserRouter,
  createRoutesFromElements,
  Route,
  RouterProvider,
} from 'react-router-dom';
import '@/index.css';

import { ToastProvider } from './context/ToastContext.jsx';
import { ApiProvider } from './context/ApiContext.jsx';
import { TradingModeProvider } from './context/TradingmodeContext.jsx';
import { ToolProvider } from './context/ToolContext';

// Pages/Layout
import App from './App.jsx';
import DashboardScreen from './routes/DashboardScreen/DashboardScreen.jsx';
import NotFoundScreen from './routes/NotfoundScreen/NotFoundScreen.jsx';

const router = createBrowserRouter(
  createRoutesFromElements(
    <Route path="/" element={<App />}>
      <Route index element={<DashboardScreen />} />
      <Route path="*" element={<NotFoundScreen />} />
    </Route>,
  ),
);

ReactDOM.createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ToastProvider>
      <TradingModeProvider>
        <ApiProvider>
          <ToolProvider>
            <RouterProvider router={router} />
          </ToolProvider>
        </ApiProvider>
      </TradingModeProvider>
    </ToastProvider>
  </StrictMode>,
);
