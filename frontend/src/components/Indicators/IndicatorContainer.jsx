import { useRef, useEffect } from 'react';
import PropTypes from 'prop-types';
import RSIChart from './RSI';
import SMAChart from './SMA';
import EMAChart from './EMA';
import BollingerBands from './BB';
import MarkovChart from './Markov';
import styles from './styles/IndicatorContainer.module.css';

const IndicatorsContainer = ({
  marketData,
  chartRef,
  highPrice,
  lowPrice,
  timelineKeys,
  candleWidth = 8,
  keyToIndex,
}) => {
  const indicatorsRef = useRef(null);

  useEffect(() => {
    const chartElement = chartRef.current;
    const indicatorsElement = indicatorsRef.current;
    if (!chartElement || !indicatorsElement) return;
    const syncScroll = () => (indicatorsElement.scrollLeft = chartElement.scrollLeft);
    chartElement.addEventListener('scroll', syncScroll);
    return () => chartElement.removeEventListener('scroll', syncScroll);
  }, [chartRef]);

  const totalWidth = (timelineKeys?.length || marketData.length) * candleWidth;

  return (
    <div className={styles.indicatorsContainer} ref={indicatorsRef}>
      <div className={styles.indicatorContent} style={{ width: `${totalWidth}px` }}>
        <BollingerBands
          marketData={marketData}
          timelineKeys={timelineKeys}
          keyToIndex={keyToIndex}
        />
        <MarkovChart
          marketData={marketData}
          timelineKeys={timelineKeys}
          keyToIndex={keyToIndex}
        />
        <RSIChart
          marketData={marketData}
          timelineKeys={timelineKeys}
          keyToIndex={keyToIndex}
        />
        <SMAChart
          marketData={marketData}
          highPrice={highPrice}
          lowPrice={lowPrice}
          timelineKeys={timelineKeys}
          keyToIndex={keyToIndex}
        />
        <EMAChart
          marketData={marketData}
          highPrice={highPrice}
          lowPrice={lowPrice}
          timelineKeys={timelineKeys}
          keyToIndex={keyToIndex}
        />
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
