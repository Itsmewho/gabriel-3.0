export const fetchHistoricTrades = async (mode) => {
  try {
    const res = await fetch(`/api/historic?mode=${mode}`);
    const json = await res.json();

    if (json.success && Array.isArray(json.trades)) {
      return { success: true, trades: json.trades };
    } else {
      return { success: false, error: 'Invalid data format' };
    }
  } catch (error) {
    return { success: false, error: error.message || 'Unknown error' };
  }
};
