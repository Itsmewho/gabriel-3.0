/* eslint-disable jsx-a11y/label-has-associated-control */
import { useState, useEffect, useMemo, useRef } from 'react';
import TrendChart from './TrendChart';
import PriceBar from './PriceBar';
import Loader from '../../../loader';
import { useApi } from '../../context/ApiContext';
import { fetchCalendar } from '../../api/Calendar';
import windowStyles from './styles/TrendWindow.module.css';

const MS_PER_HOUR = 60 * 60 * 1000;

const minuteKey = (t) => {
  if (typeof t === 'string') return t.slice(0, 16); // assumes already local
  const d = new Date(t);
  const p2 = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p2(d.getMonth() + 1)}-${p2(d.getDate())}T${p2(
    d.getHours(),
  )}:${p2(d.getMinutes())}`;
};

export default function TrendWindow() {
  const { trendMarketData: liveData } = useApi();

  const [history, setHistory] = useState([]);
  const historyMapRef = useRef(new Map());
  const [loadedOnce, setLoadedOnce] = useState(false);

  const [selectedHours, setSelectedHours] = useState(4);
  const [economicEvents, setEconomicEvents] = useState([]);

  // Fetch & incrementally merge persisted history without flicker.
  useEffect(() => {
    let cancelled = false;

    const mergeHistory = (arr) => {
      if (!Array.isArray(arr) || arr.length === 0) return false;
      let changed = false;
      const map = historyMapRef.current;
      for (const b of arr) {
        if (!b || !b.time) continue;
        const k = minuteKey(b.time);
        const prev = map.get(k);
        // Only update if materially different
        if (
          !prev ||
          prev.open !== b.open ||
          prev.high !== b.high ||
          prev.low !== b.low ||
          prev.close !== b.close ||
          prev.tick_volume !== b.tick_volume
        ) {
          map.set(k, b);
          changed = true;
        }
      }
      if (changed) {
        const next = Array.from(map.values()).sort(
          (a, b) => new Date(a.time) - new Date(b.time),
        );
        setHistory(next);
      }
      return changed;
    };

    const fetchOnce = async () => {
      try {
        const res = await fetch('/api/history/EURUSD/1m');
        const json = await res.json().catch(() => null);
        if (!json || json.success !== true) return;
        // Ignore empty refreshes if shown data
        if (json.data && json.data.length) mergeHistory(json.data);
        if (!cancelled) setLoadedOnce(true);
      } catch {}
    };

    fetchOnce();
    const id = setInterval(fetchOnce, 15000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  // Merge history + last live candle (dedupe by minute)
  const merged = useMemo(() => {
    const map = new Map();
    for (const b of history) if (b?.time) map.set(minuteKey(b.time), b);
    const last = liveData?.length ? liveData[liveData.length - 1] : null;
    if (last?.time) map.set(minuteKey(last.time), last);
    return Array.from(map.values()).sort((a, b) => new Date(a.time) - new Date(b.time));
  }, [history, liveData]);

  const isLoading = !loadedOnce && merged.length === 0;

  // Window filter
  const filteredData = useMemo(() => {
    if (!merged.length) return [];
    const last = new Date(merged[merged.length - 1].time);
    const cutoff = new Date(last.getTime() - selectedHours * MS_PER_HOUR);
    return merged.filter((e) => new Date(e.time) >= cutoff);
  }, [merged, selectedHours]);

  // Shared price scale
  const scale = useMemo(() => {
    if (!filteredData.length) return { high: 0, low: 0 };
    let hi = -Infinity,
      lo = Infinity;
    for (const c of filteredData) {
      if (c.high != null) hi = Math.max(hi, c.high);
      if (c.low != null) lo = Math.min(lo, c.low);
    }
    if (!isFinite(hi) || !isFinite(lo)) return { high: 0, low: 0 };
    return { high: hi, low: lo };
  }, [filteredData]);

  useEffect(() => {
    const run = async () => {
      const today = new Date();
      const dates = [-4, -3, -2, -1, 0, 1].map((d) => {
        const t = new Date(today);
        t.setDate(today.getDate() + d);
        return t.toISOString().slice(0, 10);
      });
      const results = await Promise.allSettled(dates.map((d) => fetchCalendar(d)));
      const all = [];
      results.forEach((res, i) => {
        if (res.status === 'fulfilled' && res.value?.success) {
          const dateStr = dates[i];
          const list = res.value.data.map((e) => ({
            ...e,
            fullEventDate: new Date(`${dateStr}T${e.time || '00:00'}:00`),
          }));
          all.push(...list);
        }
      });
      const unique = Array.from(
        new Map(all.map((e) => [e.id || `${e.fullEventDate}-${e.event}`, e])).values(),
      );
      setEconomicEvents(unique);
    };
    run();
  }, []);

  const currentPrice = filteredData.length
    ? filteredData[filteredData.length - 1].close || 0
    : 0;

  return (
    <div className={windowStyles.trendWindow}>
      {isLoading ? (
        <Loader />
      ) : (
        <>
          <div className={windowStyles.trend_container}>
            <TrendChart
              data={filteredData}
              scale={scale}
              economicEvents={economicEvents}
            />
          </div>
          <div className={windowStyles.price_bar_container}>
            <div className={windowStyles.spacer}>
              <input
                type="number"
                min="1"
                max="144"
                step="1"
                value={selectedHours}
                onChange={(e) =>
                  setSelectedHours(Math.max(1, Math.min(144, Number(e.target.value))))
                }
                className={windowStyles.input}
              />
            </div>
            <PriceBar
              marketData={filteredData}
              currentPrice={currentPrice}
              selectedHours={selectedHours}
            />
            <div className={windowStyles.spacer} />
          </div>
        </>
      )}
    </div>
  );
}
