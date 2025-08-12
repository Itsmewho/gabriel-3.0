// Fetch all open orders (paper or live)
export const fetchOpenOrders = async (mode) => {
  const endpoint = mode === 'paper' ? '/api/paper/open' : '/api/open_orders';
  try {
    const res = await fetch(endpoint);
    const data = await res.json();
    if (Array.isArray(data.orders)) {
      return { success: true, orders: data.orders };
    }
    return { success: false, error: data?.error || 'No open orders found' };
  } catch {
    return { success: false, error: 'Network error while fetching open orders' };
  }
};

// Modify a specific order (paper or live)
export const modifyOrder = async (mode, orderId, payload) => {
  const endpoint =
    mode === 'paper' ? `/api/paper/modify/${orderId}` : `/api/modify_order/${orderId}`;
  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return res.json();
  } catch {
    return { success: false, error: 'Failed to modify order' };
  }
};

// Close a specific order (paper or live)
export const closeOrder = async (mode, orderId) => {
  const endpoint =
    mode === 'paper' ? `/api/paper/close/${orderId}` : `/api/close_order/${orderId}`;
  try {
    const res = await fetch(endpoint, { method: 'POST' });
    return res.json();
  } catch {
    return { success: false, error: 'Failed to close order' };
  }
};

// Close all live orders (no paper equivalent here)
export const closeAllOrders = async () => {
  try {
    const res = await fetch('/api/close_all', { method: 'POST' });
    return res.json();
  } catch {
    return { success: false, error: 'Failed to close all orders' };
  }
};

// Place a new market order (paper or live)
export const placeOrder = async (mode, payload) => {
  const endpoint = mode === 'paper' ? '/api/paper/place' : '/api/place_market';
  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return res.json();
  } catch {
    return { success: false, error: 'Failed to place order' };
  }
};

// Fetch current symbol tick info
export const fetchSymbolTick = async (symbol) => {
  try {
    const res = await fetch(`/api/symbol_tick/${symbol}`);
    return res.json();
  } catch {
    return { success: false, error: 'Failed to fetch symbol tick' };
  }
};

// Fetch account info (paper or live)
export const fetchAccountInfo = async (mode) => {
  const endpoint = mode === 'paper' ? '/api/paper/account_info' : '/api/account_info';
  try {
    const res = await fetch(endpoint);
    return res.json();
  } catch {
    return { success: false, error: 'Failed to fetch account info' };
  }
};
