/* eslint-disable jsx-a11y/no-noninteractive-tabindex */
/* eslint-disable jsx-a11y/no-noninteractive-element-interactions */
import React, { useEffect, useState, useRef, useMemo, useCallback } from 'react';
import PropTypes from 'prop-types';
import { useToolContext } from '../../context/ToolContext';
import TooltipCandle from '../ToolTip/TooltipCandle';
import IndicatorsContainer from '../Indicators/IndicatorContainer';
import EventMarker from '../EventsMarker/EventMarker';
import chartStyles from './styles/TrendChart.module.css';

const minuteKey = (t) => {
  if (typeof t === 'string') return t.slice(0, 16);
  const d = new Date(t);
  const p2 = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p2(d.getMonth() + 1)}-${p2(d.getDate())}T${p2(
    d.getHours(),
  )}:${p2(d.getMinutes())}`;
};

const parseLocalMinute = (s) => {
  const [d, hm] = s.split('T');
  const [Y, M, D] = d.split('-').map(Number);
  const [h, m] = hm.split(':').map(Number);
  return new Date(Y, M - 1, D, h, m, 0, 0);
};

const fmtLocalMinute = (d) => {
  const p2 = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p2(d.getMonth() + 1)}-${p2(d.getDate())}T${p2(
    d.getHours(),
  )}:${p2(d.getMinutes())}`;
};

// Helper: skip Saturday(6) and Sunday(0)
const isWeekend = (d) => {
  const day = d.getDay();
  return day === 0 || day === 6;
};

// Build a dense, gapless timeline that EXCLUDES weekend minutes
const buildTimeline = (data, futureMinutes = 60) => {
  if (!data?.length) return [];
  const first = minuteKey(data[0].time);
  const last = minuteKey(data[data.length - 1].time);
  const start = parseLocalMinute(first);
  const end = parseLocalMinute(last);
  const endPlus = new Date(end.getTime() + futureMinutes * 60000);
  const out = [];
  for (let t = start.getTime(); t <= endPlus.getTime(); t += 60000) {
    const d = new Date(t);
    if (isWeekend(d)) continue; // skip weekends to avoid empty slots/gaps
    out.push(fmtLocalMinute(d));
  }
  return out;
};

export const TrendChart = ({
  data = [],
  scale,
  economicEvents = [],
  candleWidth = 8,
}) => {
  const chartRef = useRef(null);
  const { isCrosshairActive, setIsCrosshairActive } = useToolContext();

  const [crosshair, setCrosshair] = useState({ x: null, y: null, visible: false });
  const [hoveredCandle, setHoveredCandle] = useState(null);
  const [userScrolled, setUserScrolled] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [startX, setStartX] = useState(0);
  const [scrollLeft, setScrollLeft] = useState(0);

  const dataDedup = useMemo(() => {
    const m = new Map();
    for (const c of data) if (c?.time) m.set(minuteKey(c.time), c);
    return Array.from(m.values()).sort((a, b) => new Date(a.time) - new Date(b.time));
  }, [data]);

  const high = scale?.high ?? 0;
  const low = scale?.low ?? 0;
  const range = Math.max(1e-9, high - low);

  // Timeline that skips weekends
  const minuteKeys = useMemo(() => buildTimeline(dataDedup, 60), [dataDedup]);

  const keyToIndex = useMemo(
    () => new Map(minuteKeys.map((k, i) => [k, i])),
    [minuteKeys],
  );

  const dataByMinute = useMemo(() => {
    const m = new Map();
    for (const c of dataDedup) m.set(minuteKey(c.time), c);
    return m;
  }, [dataDedup]);

  const eventsByIndex = useMemo(() => {
    if (!economicEvents?.length || !minuteKeys.length) return new Map();
    const timeToIndex = new Map(minuteKeys.map((k, i) => [k, i]));
    const grouped = new Map();
    const order = { High: 0, Medium: 1, Low: 2 };
    for (const ev of economicEvents) {
      if (!ev.fullEventDate) continue;
      const k = minuteKey(ev.fullEventDate);
      const idx = timeToIndex.get(k);
      if (idx == null) continue;
      if (!grouped.has(idx)) grouped.set(idx, []);
      const arr = grouped.get(idx);
      arr.push(ev);
      arr.sort((a, b) => (order[a.impact] ?? 9) - (order[b.impact] ?? 9));
    }
    return grouped;
  }, [economicEvents, minuteKeys]);

  // Auto-scroll only when timeline grows
  const prevLenRef = useRef(0);
  useEffect(() => {
    if (!chartRef.current) return;
    if (userScrolled) {
      prevLenRef.current = minuteKeys.length;
      return;
    }
    if (minuteKeys.length > prevLenRef.current) {
      prevLenRef.current = minuteKeys.length;
      const el = chartRef.current;
      const id = requestAnimationFrame(() => {
        el.scrollLeft = el.scrollWidth;
      });
      return () => cancelAnimationFrame(id);
    }
    prevLenRef.current = minuteKeys.length;
  }, [minuteKeys, userScrolled]);

  const handleUserScroll = useCallback(() => {
    if (!userScrolled) setUserScrolled(true);
  }, [userScrolled]);

  useEffect(() => {
    const chartEl = chartRef.current;
    if (!chartEl) return;
    const onWheel = (e) => {
      requestAnimationFrame(() => {
        chartEl.scrollLeft += e.deltaX * 2;
        e.preventDefault();
        handleUserScroll();
      });
    };
    chartEl.addEventListener('wheel', onWheel, { passive: false });
    return () => chartEl.removeEventListener('wheel', onWheel);
  }, [handleUserScroll]);

  const onMouseDown = (e) => {
    setIsDragging(true);
    const el = chartRef.current;
    if (!el) return;
    setStartX(e.pageX - el.offsetLeft);
    setScrollLeft(el.scrollLeft);
  };
  const onMouseMoveDrag = (e) => {
    if (!isDragging || !chartRef.current) return;
    e.preventDefault();
    const x = e.pageX - chartRef.current.offsetLeft;
    chartRef.current.scrollLeft = scrollLeft - (x - startX) * 2;
  };
  const onMouseUp = () => setIsDragging(false);
  const onMouseLeaveDrag = () => setIsDragging(false);

  const onMouseMoveCross = (e) => {
    if (!isCrosshairActive || !chartRef.current) return;
    const rect = chartRef.current.getBoundingClientRect();
    let x = e.clientX - rect.left + chartRef.current.scrollLeft;
    let y = e.clientY - rect.top;
    x = Math.max(0, Math.min(x, chartRef.current.scrollWidth));
    y = Math.max(0, Math.min(y, rect.height));
    let index = Math.floor(x / candleWidth);
    index = Math.max(0, Math.min(index, minuteKeys.length - 1));
    const key = minuteKeys[index];
    const c = dataByMinute.get(key);
    if (c) {
      const cx = index * candleWidth + candleWidth / 2 - chartRef.current.scrollLeft;
      setCrosshair({ x: cx, y, visible: true });
      setHoveredCandle(c);
    }
  };

  const onMouseLeaveCross = () => {
    setCrosshair({ x: null, y: null, visible: false });
    setHoveredCandle(null);
  };

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') setIsCrosshairActive(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setIsCrosshairActive]);

  const labelHHMM = useCallback((k) => k.slice(11, 16), []);

  return (
    <div
      className={chartStyles.trend_chart_container}
      onMouseMove={onMouseMoveCross}
      onMouseLeave={onMouseLeaveCross}
      style={{ cursor: isCrosshairActive ? 'crosshair' : 'default' }}
    >
      <div className={chartStyles.chart_ticker}>
        <div className={chartStyles.ticker_flex}>EURUSD</div>
      </div>
      <div
        className={`${chartStyles.chart_container} ${
          isDragging ? chartStyles.dragging : ''
        }`}
        ref={chartRef}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMoveDrag}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseLeaveDrag}
        tabIndex={0}
        role="region"
        aria-label="Trend chart scroll area"
      >
        <div className={chartStyles.candle_container}>
          {minuteKeys.map((k) => {
            const c = dataByMinute.get(k);
            return c ? (
              <Candle
                key={`c-${k}`}
                candle={c}
                high={high}
                low={low}
                range={range}
                width={candleWidth}
              />
            ) : (
              <div
                key={`s-${k}`}
                style={{ width: candleWidth }}
                className={chartStyles.candleSlot}
              />
            );
          })}
          {isCrosshairActive && crosshair.visible && (
            <div className={chartStyles.crosshair_overlay}>
              <div
                className={chartStyles.crosshair_v}
                style={{ left: `${crosshair.x}px` }}
              />
              <div
                className={chartStyles.crosshair_h}
                style={{ top: `${crosshair.y}px` }}
              />
            </div>
          )}
          {hoveredCandle && isCrosshairActive && crosshair.visible && (
            <TooltipCandle
              x={crosshair.x}
              y={crosshair.y}
              candleData={hoveredCandle}
              containerRef={chartRef}
            />
          )}
        </div>
        <div className={chartStyles.timeline_container}>
          <div className={chartStyles.times}>
            {minuteKeys.map((k, idx) => (
              <div
                key={`t-${k}`}
                className={
                  idx % 10 === 0 ? chartStyles.visibleTime : chartStyles.hiddenTime
                }
              >
                {labelHHMM(k)}
              </div>
            ))}
          </div>
          <div className={chartStyles.events_overlay}>
            {Array.from(eventsByIndex.entries()).map(([index, events]) => (
              <div
                key={index}
                className={chartStyles.eventStack}
                style={{ left: `${index * candleWidth + candleWidth / 2}px` }}
              >
                {events.map((ev) => (
                  <EventMarker key={ev.id || `${ev.event}-${ev.time}`} event={ev} />
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>
      <IndicatorsContainer
        marketData={dataDedup}
        chartRef={chartRef}
        highPrice={high}
        lowPrice={low}
        timelineKeys={minuteKeys}
        candleWidth={candleWidth}
        keyToIndex={keyToIndex}
      />
    </div>
  );
};

TrendChart.propTypes = {
  data: PropTypes.arrayOf(PropTypes.object).isRequired,
  scale: PropTypes.shape({ high: PropTypes.number, low: PropTypes.number }).isRequired,
  economicEvents: PropTypes.arrayOf(PropTypes.object),
  candleWidth: PropTypes.number,
};

const Candle = React.memo(({ candle, high, low, range, width }) => {
  const open = candle.open ?? 0;
  const close = candle.close ?? 0;
  const highP = candle.high ?? 0;
  const lowP = candle.low ?? 0;
  const bodyHeight = Math.abs(close - open);
  const wickHeight = highP - lowP;
  return (
    <div className={chartStyles.candle} style={{ width }}>
      <div
        className={chartStyles.wick}
        style={{
          height: `${(wickHeight / range) * 100}%`,
          bottom: `${((lowP - low) / range) * 100}%`,
        }}
      />
      <div
        className={`${chartStyles.candleBody} ${
          close > open ? chartStyles.greenCandle : chartStyles.redCandle
        }`}
        style={{
          height: `${(bodyHeight / range) * 100}%`,
          bottom: `${((Math.min(open, close) - low) / range) * 100}%`,
        }}
      />
    </div>
  );
});
Candle.displayName = 'Candle';
Candle.propTypes = {
  candle: PropTypes.shape({
    time: PropTypes.string.isRequired,
    open: PropTypes.number,
    close: PropTypes.number,
    high: PropTypes.number,
    low: PropTypes.number,
  }).isRequired,
  high: PropTypes.number.isRequired,
  low: PropTypes.number.isRequired,
  range: PropTypes.number.isRequired,
  width: PropTypes.number,
};
export default TrendChart;
