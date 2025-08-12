import { useMemo, useCallback } from 'react';
import PropTypes from 'prop-types';
import priceBarStyles from './styles/PriceBar.module.css';

export function PriceBar({ currentPrice = 0, scale }) {
  const { high = 0, low = 0 } = scale || {};
  const range = Math.max(0, high - low);

  const steps = useMemo(() => {
    if (!range) return [];
    const step = range / 10;
    return Array.from({ length: 10 }, (_, i) => (low + i * step).toFixed(5)).reverse();
  }, [low, range]);

  const pos = useCallback(
    (p) => {
      if (!range || p == null) return '50%';
      return `${((high - p) / range) * 100}%`;
    },
    [high, range],
  );

  return (
    <div className={priceBarStyles.price_bar_container}>
      <div className={priceBarStyles.price_labels}>
        <div className={priceBarStyles.high_label} style={{ top: pos(high) }}>
          {high.toFixed?.(5) ?? '0.00000'}
        </div>

        {steps.map((s, i) => (
          <div key={i} className={priceBarStyles.price_step}>
            {s}
          </div>
        ))}

        <div className={priceBarStyles.current_label} style={{ top: pos(currentPrice) }}>
          {currentPrice.toFixed(5)}
        </div>

        <div className={priceBarStyles.low_label} style={{ top: pos(low) }}>
          {low.toFixed?.(5) ?? '0.00000'}
        </div>
      </div>
    </div>
  );
}

PriceBar.propTypes = {
  currentPrice: PropTypes.number.isRequired,
  scale: PropTypes.shape({ high: PropTypes.number, low: PropTypes.number }).isRequired,
};

export default PriceBar;
