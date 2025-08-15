// utils/AlignedInterval.js
/**
 * Starts an interval aligned to the top of each minute + `offsetSec`.
 * Runs the callback immediately once to avoid long cold starts.
 *
 * @param {Function} callback
 * @param {number} intervalMs
 * @param {number} offsetSec
 * @returns {Function} cleanup
 */
export function startAlignedInterval(callback, intervalMs, offsetSec = 1) {
  let intervalId;

  const safeRun = () => {
    try {
      callback();
    } catch (_) {}
  };

  // Fast first paint: run immediately
  safeRun();

  // Schedule the first aligned tick, then regular interval
  const now = new Date();
  const next = new Date(
    now.getFullYear(),
    now.getMonth(),
    now.getDate(),
    now.getHours(),
    now.getMinutes() + 1,
    offsetSec,
    0,
  );
  const delay = Math.max(0, next.getTime() - now.getTime());

  const timeoutId = setTimeout(() => {
    safeRun();
    intervalId = setInterval(safeRun, intervalMs);
  }, delay);

  return () => {
    clearTimeout(timeoutId);
    if (intervalId) clearInterval(intervalId);
  };
}
