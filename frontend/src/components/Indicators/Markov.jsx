import { useMemo } from 'react';
import PropTypes from 'prop-types';
import styles from './styles/Markov.module.css';
import { useApi } from '../../context/ApiContext';

const MarkovChart = ({ marketData }) => {
  const { markovHistory } = useApi();

  const alignedMarkovData = useMemo(() => {
    if (!Array.isArray(markovHistory) || !markovHistory.length || !marketData.length)
      return [];

    const marketTimestamps = new Set(
      marketData.map((entry) => entry.time.substring(0, 19)),
    );

    return markovHistory.filter((state) => {
      const stateTimestamp = state.time.substring(0, 19);
      return marketTimestamps.has(stateTimestamp);
    });
  }, [markovHistory, marketData]);

  return (
    <div className={styles.markovContainer}>
      <div className={styles.markovChart}>
        {alignedMarkovData.length > 0 ? (
          alignedMarkovData.map((state, idx) => (
            <div
              key={idx}
              className={`${styles.markovBar} ${
                state.state === 'Bullish'
                  ? styles.bullish
                  : state.state === 'Bearish'
                    ? styles.bearish
                    : styles.neutral
              }`}
              style={{
                height:
                  state.state === 'Bullish'
                    ? '50%'
                    : state.state === 'Bearish'
                      ? '40%'
                      : '15%',
                left: `${(idx / alignedMarkovData.length) * 100}%`,
              }}
            />
          ))
        ) : (
          <p className={styles.noData}>Markov</p>
        )}
      </div>
    </div>
  );
};

MarkovChart.propTypes = {
  marketData: PropTypes.arrayOf(PropTypes.object).isRequired,
};

export default MarkovChart;
