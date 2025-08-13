// context/ApiContext.jsx
import PropTypes from 'prop-types';
import {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
  useRef,
} from 'react';
import { FETCHTIME, LASTCANDLETIME } from '../config/constants';
import { startAlignedInterval } from '../utils/AlignedInterval';
import { fetchHistoricTrades } from '../api/Historic';
import { fetchMarkovData } from '../api/Data';
import { fetchTrendEvaluation } from '../api/TrendEval';
import { useTradingMode } from '../context/TradingmodeContext';
import {
  fetchSMA,
  fetchRSI,
  fetchMarkovHistory,
  fetchEMA,
  fetchBollingerData,
} from '../api/Indicators';
import { fetchLastCandle, fetchMarketHistory } from '../api/Trend';

const ApiContext = createContext(null);

const minuteKey = (iso) => {
  const d = new Date(iso);
  d.setSeconds(0, 0);
  return d.toISOString();
};

const dedupeByMinuteSorted = (arr = []) => {
  if (!Array.isArray(arr) || arr.length === 0) return [];
  const map = new Map();
  for (const c of arr) {
    if (!c?.time) continue;
    map.set(minuteKey(c.time), c); // last wins
  }
  return Array.from(map.entries())
    .sort((a, b) => (a[0] < b[0] ? -1 : 1))
    .map(([, v]) => v);
};

export function ApiProvider({ children }) {
  const [emaData, setEmaData] = useState([]);
  const [smaData, setSmaData] = useState([]);
  const [rsiData, setRsiData] = useState([]);
  const [markovHistory, setMarkovHistory] = useState([]);
  const [markovData, setMarkovData] = useState(null);
  const [evalData, setEvalData] = useState(null);
  const [historicTrades, setHistoricTrades] = useState([]);
  const [historicError, setHistoricError] = useState(null);
  const [trendMarketData, setTrendMarketData] = useState([]);
  const [bollingerData, setBollingerData] = useState([]);
  const { mode } = useTradingMode();
  const hasHistoryRef = useRef(false);

  const fetchAllInitialData = useCallback(async () => {
    const symbol = 'EURUSD';
    const timeframe = '1m';

    const results = await Promise.allSettled([
      fetchMarketHistory(symbol, timeframe),
      fetchEMA(),
      fetchSMA(),
      fetchRSI(),
      fetchMarkovHistory(),
      fetchBollingerData(symbol, timeframe),
      fetchMarkovData(symbol, timeframe),
      fetchTrendEvaluation(),
      fetchHistoricTrades(mode),
    ]);

    const [
      historyRes,
      emaRes,
      smaRes,
      rsiRes,
      markovHistRes,
      bollingerRes,
      markovDataRes,
      evalRes,
      tradesRes,
    ] = results;

    if (historyRes.status === 'fulfilled' && historyRes.value.success) {
      hasHistoryRef.current = true;
      setTrendMarketData(dedupeByMinuteSorted(historyRes.value.data));
    }
    if (emaRes.status === 'fulfilled' && emaRes.value.success)
      setEmaData(emaRes.value.data);
    if (smaRes.status === 'fulfilled' && smaRes.value.success)
      setSmaData(smaRes.value.data);
    if (rsiRes.status === 'fulfilled' && rsiRes.value.success)
      setRsiData(rsiRes.value.data);
    if (markovHistRes.status === 'fulfilled' && markovHistRes.value.success)
      setMarkovHistory(markovHistRes.value.data);
    if (bollingerRes.status === 'fulfilled' && bollingerRes.value.success)
      setBollingerData(bollingerRes.value.data);
    if (markovDataRes.status === 'fulfilled' && markovDataRes.value.success)
      setMarkovData(markovDataRes.value.data);
    if (evalRes.status === 'fulfilled' && evalRes.value.success)
      setEvalData(evalRes.value.data);

    if (tradesRes.status === 'fulfilled' && tradesRes.value.success) {
      setHistoricTrades(tradesRes.value.trades);
      setHistoricError(null);
    } else if (tradesRes.status === 'fulfilled') {
      setHistoricTrades([]);
      setHistoricError(tradesRes.value.error || 'Failed to fetch trade data.');
    }
  }, [mode]);

  useEffect(() => {
    fetchAllInitialData();

    const id = setInterval(fetchAllInitialData, FETCHTIME);

    const cleanupLastCandle = startAlignedInterval(
      async () => {
        const symbol = 'EURUSD';
        const timeframe = '1m';
        const lastCandleRes = await fetchLastCandle(symbol, timeframe);
        if (!lastCandleRes.success) return;

        setTrendMarketData((prev) => {
          if (!hasHistoryRef.current) {
            return lastCandleRes.data ? [lastCandleRes.data] : [];
          }

          const next = Array.isArray(prev) ? [...prev] : [];
          const k = minuteKey(lastCandleRes.data.time);
          const idx = next.findIndex((c) => c?.time && minuteKey(c.time) === k);
          if (idx !== -1) next[idx] = lastCandleRes.data;
          else next.push(lastCandleRes.data);

          const MAX_KEEP = 1; // cap growth
          if (next.length > MAX_KEEP) next.splice(0, next.length - MAX_KEEP);

          next.sort((a, b) => new Date(a.time) - new Date(b.time));
          return next;
        });
      },
      LASTCANDLETIME,
      2,
    );

    return () => {
      clearInterval(id);
      cleanupLastCandle();
    };
  }, [fetchAllInitialData]);

  const value = {
    markovData,
    evalData,
    historicTrades,
    historicError,
    emaData,
    smaData,
    rsiData,
    markovHistory,
    trendMarketData,
    bollingerData,
  };

  return <ApiContext.Provider value={value}>{children}</ApiContext.Provider>;
}

ApiProvider.propTypes = { children: PropTypes.node.isRequired };

export function useApi() {
  const ctx = useContext(ApiContext);
  if (!ctx) throw new Error('useApi must be used within ApiProvider');
  return ctx;
}
