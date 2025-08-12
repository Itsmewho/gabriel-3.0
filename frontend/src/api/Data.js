export const fetchMarkovData = async (symbol = 'EURUSD', timeframe = '1m') => {
  try {
    const res = await fetch(`/api/markov/${symbol}/${timeframe}`);
    const json = await res.json();

    if (json.success && json.data) {
      return { success: true, data: json.data };
    } else {
      return { success: false, error: 'Invalid Markov data' };
    }
  } catch (error) {
    return { success: false, error: error.message || 'Failed to fetch Markov data' };
  }
};
