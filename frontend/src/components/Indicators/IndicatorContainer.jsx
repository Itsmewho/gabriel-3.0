import { useRef, useEffect } from 'react';
import PropTypes from 'prop-types';
import RSIChart from './RSI';
import SMAChart from './SMA';
import EMAChart from './EMA';
import BollingerBands from './BB';
import MarkovChart from './Markov';
import styles from './styles/IndicatorContainer.module.css';
const IndicatorsContainer = ({ marketData, chartRef, highPrice, lowPrice }) => {
  const indicatorsRef = useRef(null);

  useEffect(() => {
    const chartElement = chartRef.current;
    const indicatorsElement = indicatorsRef.current;

    if (chartElement && indicatorsElement) {
      const syncScroll = () => {
        indicatorsElement.scrollLeft = chartElement.scrollLeft;
      };

      chartElement.addEventListener('scroll', syncScroll);
      return () => chartElement.removeEventListener('scroll', syncScroll);
    }
  }, [chartRef]);

  return (
    <div className={styles.indicatorsContainer} ref={indicatorsRef}>
      <div
        className={styles.indicatorContent}
        style={{ width: `${marketData.length * 8}px` }} // Set width for scrolling
      >
        <BollingerBands marketData={marketData} />
        <MarkovChart marketData={marketData} />
        <RSIChart marketData={marketData} />
        <SMAChart marketData={marketData} highPrice={highPrice} lowPrice={lowPrice} />
        <EMAChart marketData={marketData} highPrice={highPrice} lowPrice={lowPrice} />
      </div>
    </div>
  );
};

IndicatorsContainer.propTypes = {
  marketData: PropTypes.arrayOf(PropTypes.object).isRequired,
  chartRef: PropTypes.shape({ current: PropTypes.instanceOf(Element) }).isRequired,
  highPrice: PropTypes.number.isRequired,
  lowPrice: PropTypes.number.isRequired,
};

export default IndicatorsContainer;
