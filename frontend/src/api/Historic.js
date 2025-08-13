export const fetchHistoricTrades = async (mode) => {
  try {
    const res = await fetch(`/api/historic?mode=${encodeURIComponent(mode)}`);
    const json = await res.json().catch(() => null);

    if (!res.ok || !json || json.success !== true || !Array.isArray(json.trades)) {
      return { success: false, error: (json && json.error) || `HTTP ${res.status}` };
    }

    return { success: true, trades: json.trades, account: json.account || null };
  } catch (error) {
    return { success: false, error: error?.message || 'Unknown error' };
  }
};
