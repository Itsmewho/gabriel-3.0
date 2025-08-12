export async function fetchCalendar(date) {
  try {
    const res = await fetch(`/api/calendar?date=${date}`);
    const data = await res.json();

    if (Array.isArray(data)) {
      return { success: true, data };
    }

    return {
      success: false,
      message: data?.error || 'No calendar data available',
    };
  } catch {
    return {
      success: false,
      message: 'Network error while fetching calendar data',
    };
  }
}
