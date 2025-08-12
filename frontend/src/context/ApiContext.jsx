// context/ApiContext.jsx
import PropTypes from 'prop-types';
import { FETCHTIME, LASTCANDLETIME } from '../config/constants';
import { createContext, useContext, useEffect, useState, useCallback } from 'react';
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

const ApiContext = createContext();

export const ApiProvider = ({ children }) => {
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
      setTrendMarketData(historyRes.value.data);
    }
    if (emaRes.status === 'fulfilled' && emaRes.value.success) {
      setEmaData(emaRes.value.data);
    }
    if (smaRes.status === 'fulfilled' && smaRes.value.success) {
      setSmaData(smaRes.value.data);
    }
    if (rsiRes.status === 'fulfilled' && rsiRes.value.success) {
      setRsiData(rsiRes.value.data);
    }
    if (markovHistRes.status === 'fulfilled' && markovHistRes.value.success) {
      setMarkovHistory(markovHistRes.value.data);
    }
    if (bollingerRes.status === 'fulfilled' && bollingerRes.value.success) {
      setBollingerData(bollingerRes.value.data);
    }
    if (markovDataRes.status === 'fulfilled' && markovDataRes.value.success) {
      setMarkovData(markovDataRes.value.data);
    }
    if (evalRes.status === 'fulfilled' && evalRes.value.success) {
      setEvalData(evalRes.value.data);
    }
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

    const intervalId = setInterval(fetchAllInitialData, FETCHTIME);

    const cleanupLastCandle = startAlignedInterval(
      async () => {
        const symbol = 'EURUSD';
        const timeframe = '1m';
        const lastCandleRes = await fetchLastCandle(symbol, timeframe);
        if (lastCandleRes.success) {
          setTrendMarketData((prev) => {
            if (!prev || prev.length === 0) return [lastCandleRes.data];

            const existingIndex = prev.findIndex(
              (c) => c.time === lastCandleRes.data.time,
            );

            if (existingIndex !== -1) {
              const updatedData = [...prev];
              updatedData[existingIndex] = lastCandleRes.data;
              return updatedData;
            } else {
              return [...prev, lastCandleRes.data];
            }
          });
        }
      },
      LASTCANDLETIME,
      2,
    );

    // Cleanup bij unmount
    return () => {
      clearInterval(intervalId);
      cleanupLastCandle();
    };
  }, [fetchAllInitialData]);

  return (
    <ApiContext.Provider
      value={{
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
      }}
    >
      {children}
    </ApiContext.Provider>
  );
};

ApiProvider.propTypes = {
  children: PropTypes.node.isRequired,
};

export const useApi = () => useContext(ApiContext);
