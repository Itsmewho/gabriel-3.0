/* eslint-disable jsx-a11y/no-noninteractive-tabindex */
/* eslint-disable jsx-a11y/no-noninteractive-element-interactions */
import React, { useEffect, useState, useRef, useMemo, useCallback } from 'react';
import PropTypes from 'prop-types';
import { useToolContext } from '../../context/ToolContext';
import TooltipCandle from '../ToolTip/TooltipCandle';
import IndicatorsContainer from '../Indicators/IndicatorContainer';
import EventMarker from '../EventsMarker/EventMarker';
import chartStyles from './styles/TrendChart.module.css';

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

  const high = scale?.high ?? 0;
  const low = scale?.low ?? 0;
  const range = Math.max(1e-9, high - low);

  // Group events by candle index minute-aligned
  const eventsByCandleIndex = useMemo(() => {
    if (!economicEvents.length || !data.length) return new Map();

    const timeToIndex = new Map(
      data.map((candle, idx) => {
        const t = new Date(candle.time);
        t.setSeconds(0, 0);
        return [t.getTime(), idx];
      }),
    );

    const grouped = new Map();
    const order = { High: 0, Medium: 1, Low: 2 };

    for (const ev of economicEvents) {
      if (!ev.fullEventDate) continue;
      const et = new Date(ev.fullEventDate);
      et.setSeconds(0, 0);
      const k = timeToIndex.get(et.getTime());
      if (k == null) continue;
      if (!grouped.has(k)) grouped.set(k, []);
      const arr = grouped.get(k);
      arr.push(ev);
      arr.sort((a, b) => (order[a.impact] ?? 9) - (order[b.impact] ?? 9));
    }

    return grouped;
  }, [data, economicEvents]);

  // Auto-scroll to the end when data changes unless user scrolled
  useEffect(() => {
    if (!chartRef.current || userScrolled) return;
    const el = chartRef.current;
    const id = requestAnimationFrame(() => {
      el.scrollLeft = el.scrollWidth;
    });
    return () => cancelAnimationFrame(id);
  }, [data, userScrolled]);

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
    index = Math.max(0, Math.min(index, data.length - 1));
    const c = data[index];
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
          {data.map((candle, idx) => (
            <Candle
              key={`${candle.time}-${idx}`}
              candle={candle}
              high={high}
              low={low}
              range={range}
            />
          ))}

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
            {data.map((candle, idx) => (
              <div
                key={`${candle.time}-${idx}`}
                className={
                  idx % 10 === 0 ? chartStyles.visibleTime : chartStyles.hiddenTime
                }
              >
                {new Date(candle.time).toLocaleTimeString([], {
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </div>
            ))}
          </div>

          <div className={chartStyles.events_overlay}>
            {Array.from(eventsByCandleIndex.entries()).map(([index, events]) => (
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
        marketData={data}
        chartRef={chartRef}
        highPrice={high}
        lowPrice={low}
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

const Candle = React.memo(({ candle, high, low, range }) => {
  const open = candle.open ?? 0;
  const close = candle.close ?? 0;
  const highP = candle.high ?? 0;
  const lowP = candle.low ?? 0;

  const bodyHeight = Math.abs(close - open);
  const wickHeight = highP - lowP;

  return (
    <div className={chartStyles.candle}>
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
};

export default TrendChart;
