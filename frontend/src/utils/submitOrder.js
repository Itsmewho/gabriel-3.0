import { placeOrder } from '../api/Orders';

export const submitOrder = async ({
  selectedSymbol,
  mode,
  volume,
  volumeMin,
  volumeMax,
  volumeStep,
  orderType,
  slPoints,
  tpPoints,
  stopLevelPoints,
  currentPrice,
  accountInfo,
  showToast,
  fetchAccountInfo,
  setVolume,
  setSlPoints,
  setTpPoints,
}) => {
  const vol = parseFloat(volume);
  const slDistance = parseFloat(slPoints);
  const tpDistance = parseFloat(tpPoints);

  const calculateAdjusted = (price, points, type, side) => {
    const factor = points * 0.00001;
    return type === 'buy'
      ? side === 'sl'
        ? price - factor
        : price + factor
      : side === 'sl'
        ? price + factor
        : price - factor;
  };

  if (vol < volumeMin || vol > volumeMax) {
    showToast(`Volume must be between ${volumeMin} and ${volumeMax}`, 'error');
    return;
  }

  const steps = Math.round((vol - volumeMin) / volumeStep);
  const expected = parseFloat((volumeMin + steps * volumeStep).toFixed(2));
  if (expected !== vol) {
    showToast(`Volume must align with step size of ${volumeStep}`, 'error');
    return;
  }

  if (slDistance && slDistance < stopLevelPoints) {
    showToast(`Stop loss must be at least ${stopLevelPoints} points away.`, 'error');
    return;
  }

  if (tpDistance && tpDistance < stopLevelPoints) {
    showToast(`Take profit must be at least ${stopLevelPoints} points away.`, 'error');
    return;
  }

  if (currentPrice && slDistance) {
    const slAbs = calculateAdjusted(currentPrice, slDistance, orderType, 'sl');
    if (
      (orderType === 'buy' && slAbs >= currentPrice) ||
      (orderType === 'sell' && slAbs <= currentPrice)
    ) {
      showToast('Invalid SL: Must be below price for BUY, above for SELL.', 'error');
      return;
    }
  }

  if (currentPrice && tpDistance) {
    const tpAbs = calculateAdjusted(currentPrice, tpDistance, orderType, 'tp');
    if (
      (orderType === 'buy' && tpAbs <= currentPrice) ||
      (orderType === 'sell' && tpAbs >= currentPrice)
    ) {
      showToast('Invalid TP: Must be above price for BUY, below for SELL.', 'error');
      return;
    }
  }

  if (mode === 'paper' && accountInfo && currentPrice) {
    const leverage = accountInfo.leverage || 100;
    const requiredMargin = (vol * 100000 * currentPrice) / leverage;

    if (requiredMargin > accountInfo.free_margin) {
      showToast('Not enough free margin to place order.', 'error');
      return;
    }

    if (requiredMargin > accountInfo.balance) {
      showToast('Order size exceeds account balance.', 'error');
      return;
    }
  }

  const payload = {
    symbol: selectedSymbol,
    volume: vol,
    order_type: orderType,
    stop_loss: slPoints
      ? calculateAdjusted(currentPrice, slDistance, orderType, 'sl')
      : null,
    take_profit: tpPoints
      ? calculateAdjusted(currentPrice, tpDistance, orderType, 'tp')
      : null,
  };

  try {
    const data = await placeOrder(mode, payload);
    if (data.success) {
      showToast(data.message || 'Order placed successfully!', 'success');

      // Wait for 2 seconds before resetting form
      setTimeout(async () => {
        await fetchAccountInfo();

        // Reset form inputs
        setVolume(0.1);
        setSlPoints('');
        setTpPoints('');
      }, 2000);
    } else {
      showToast(data.message || data.error || 'Failed to place order.', 'error');
    }
  } catch {
    showToast('Error placing order.', 'error');
  }
};
