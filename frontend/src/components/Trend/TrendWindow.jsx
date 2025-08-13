/* eslint-disable jsx-a11y/label-has-associated-control */
import { useState, useEffect, useMemo } from 'react';
import TrendChart from './TrendChart';
import PriceBar from './PriceBar';
import Loader from '../../../loader';
import { useApi } from '../../context/ApiContext';
import { fetchCalendar } from '../../api/Calendar';
import windowStyles from './styles/TrendWindow.module.css';

const MS_PER_HOUR = 60 * 60 * 1000;

export default function TrendWindow() {
  const { trendMarketData: marketData } = useApi();
  const [selectedHours, setSelectedHours] = useState(4); // match mt5 standard timeframe
  const [economicEvents, setEconomicEvents] = useState([]);

  const isLoading = !marketData.length;

  // Filter once for the selected window
  const filteredData = useMemo(() => {
    if (!marketData.length) return [];
    const last = new Date(marketData.at(-1).time);
    const cutoff = new Date(last.getTime() - selectedHours * MS_PER_HOUR);
    return marketData.filter((e) => new Date(e.time) >= cutoff);
  }, [marketData, selectedHours]);

  // Shared price scale from filtered window
  const scale = useMemo(() => {
    if (!filteredData.length) return { high: 0, low: 0 };
    let hi = -Infinity;
    let lo = Infinity;
    for (const c of filteredData) {
      if (c.high != null) hi = Math.max(hi, c.high);
      if (c.low != null) lo = Math.min(lo, c.low);
    }
    if (!isFinite(hi) || !isFinite(lo)) return { high: 0, low: 0 };
    return { high: hi, low: lo };
  }, [filteredData]);

  // Load events for a small date window, normalize to local time (no 'Z')
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
            fullEventDate: new Date(`${dateStr}T${e.time || '00:00'}:00`), // local time
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

  const currentPrice = filteredData.at(-1)?.close || 0;

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

            <PriceBar currentPrice={currentPrice} scale={scale} />

            <div className={windowStyles.spacer} />
          </div>
        </>
      )}
    </div>
  );
}
