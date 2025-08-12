/**
 * Starts an interval that aligns to the top of each minute + `offsetSec` (default: 2s).
 *
 * @param {Function} callback
 * @param {number} intervalMs
 * @param {number} offsetSec
 * @returns {Function} -
 */

export function startAlignedInterval(callback, intervalMs, offsetSec = 2) {
  const now = new Date();
  const currentMs = now.getTime();

  const nextMinute = new Date(
    Date.UTC(
      now.getUTCFullYear(),
      now.getUTCMonth(),
      now.getUTCDate(),
      now.getUTCHours(),
      now.getUTCMinutes() + 1,
      offsetSec, // aligned start at :02
      0
    )
  );

  const delay = nextMinute.getTime() - currentMs;

  let intervalId;
  const timeoutId = setTimeout(() => {
    callback();
    intervalId = setInterval(callback, intervalMs);
  }, delay);

  return () => {
    clearTimeout(timeoutId);
    if (intervalId) clearInterval(intervalId);
  };
}
