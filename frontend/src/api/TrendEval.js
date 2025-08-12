export async function fetchTrendEvaluation() {
  try {
    const res = await fetch('/api/trend-eval/EURUSD/1m');
    const data = await res.json();

    if (res.ok && data.success) {
      return { success: true, data: data.data };
    }
    return { success: false, message: data.error || 'Trend evaluation not available' };
  } catch {
    return { success: false, message: 'Network error fetching trend evaluation' };
  }
}
