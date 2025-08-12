export async function fetchRSI(symbol = 'EURUSD', timeframe = '1m') {
  try {
    const res = await fetch(`/api/rsi/${symbol}/${timeframe}`);
    const data = await res.json();
    if (data.success) {
      return { success: true, data: data.data };
    } else {
      return { success: false, message: data.error || 'Failed to fetch RSI' };
    }
  } catch {
    return { success: false, message: 'Network error fetching RSI' };
  }
}

export async function fetchSMA(symbol = 'EURUSD', timeframe = '1m') {
  try {
    const res = await fetch(`/api/sma/${symbol}/${timeframe}`);
    const data = await res.json();
    if (data.success) {
      return { success: true, data: data.data };
    } else {
      return { success: false, message: data.error || 'Failed to fetch SMA' };
    }
  } catch {
    return { success: false, message: 'Network error fetching SMA' };
  }
}

export async function fetchEMA(symbol = 'EURUSD', timeframe = '1m') {
  try {
    const res = await fetch(`/api/ema/${symbol}/${timeframe}`);
    const data = await res.json();
    if (data.success) {
      return { success: true, data: data.data };
    } else {
      return { success: false, message: data.error || 'Failed to fetch EMA' };
    }
  } catch {
    return { success: false, message: 'Network error fetching EMA' };
  }
}

export async function fetchMarkov(symbol = 'EURUSD', timeframe = '1m') {
  try {
    const res = await fetch(`/api/markov/${symbol}/${timeframe}`);
    const data = await res.json();
    if (data.success) {
      return { success: true, data: data.data };
    } else {
      return {
        success: false,
        message: data.message || 'Failed to fetch Markov prediction',
      };
    }
  } catch {
    return { success: false, message: 'Network error fetching Markov prediction' };
  }
}

export async function fetchMarkovHistory(symbol = 'EURUSD', timeframe = '1m') {
  try {
    const res = await fetch(`/api/markov/historic/${symbol}/${timeframe}`);
    const data = await res.json();
    if (data.success) {
      return { success: true, data: data.data };
    } else {
      return {
        success: false,
        message: data.message || 'Failed to fetch Markov history',
      };
    }
  } catch {
    return { success: false, message: 'Network error fetching Markov history' };
  }
}

export async function fetchBollingerData(symbol = 'EURUSD', timeframe = '1m') {
  try {
    const res = await fetch(`/api/bollinger/${symbol}/${timeframe}`);
    const data = await res.json();
    if (res.ok && data.success) {
      return { success: true, data: data.data };
    }
    return { success: false, message: data.error || 'Failed to fetch Bollinger Bands' };
  } catch {
    return { success: false, message: 'Network error fetching Bollinger Bands' };
  }
}
