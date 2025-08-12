export const fetchMarketHistory = async (symbol = 'EURUSD', timeframe = '1m') => {
  try {
    const res = await fetch(`/api/history/${symbol}/${timeframe}`);
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.message || 'Invalid history data');
    return { success: true, data: data.data };
  } catch (err) {
    return { success: false, error: err.message };
  }
};

export const fetchLastCandle = async (symbol = 'EURUSD', timeframe = '1m') => {
  try {
    const res = await fetch(`/api/lastcandle/${symbol}/${timeframe}`);
    const data = await res.json();
    if (!res.ok || !data.success)
      throw new Error(data.message || 'Invalid last candle data');
    return { success: true, data: data.data };
  } catch (err) {
    return { success: false, error: err.message };
  }
};
