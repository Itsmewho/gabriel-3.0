import styles from './styles/Data.module.css';
import { useState, useEffect } from 'react';
import { fetchCalendar } from '../../api/Calendar';

const Calendar = () => {
  const [fadeIn, setFadeIn] = useState(false);
  const [expandedIndex, setExpandedIndex] = useState(null);
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().slice(0, 10));
  const [calendarEvents, setCalendarEvents] = useState([]);

  useEffect(() => {
    setFadeIn(true);
    const load = async () => {
      const res = await fetchCalendar(selectedDate);
      if (res.success) {
        const normalized = res.data.map((event) => ({
          ...event,
          fullEventDate: new Date(`${selectedDate}T${event.time || '00:00'}:00Z`),
        }));
        setCalendarEvents(normalized);
      } else {
        setCalendarEvents([]);
      }
    };
    load();
  }, [selectedDate]);

  const toggleAccordion = (index) => {
    setExpandedIndex(expandedIndex === index ? null : index);
  };

  return (
    <div className={`${styles.view} ${fadeIn ? styles.fadeIn : ''}`}>
      <div className={styles.date_selector}>
        <label htmlFor="date">Select Date:</label>
        <input
          type="date"
          id="date"
          value={selectedDate}
          onChange={(e) => setSelectedDate(e.target.value)}
        />
      </div>

      {calendarEvents.length === 0 ? (
        <p className={styles.titles}>No events found for this date.</p>
      ) : (
        <div className={styles.accordion}>
          {calendarEvents.map((event, index) => (
            <div key={index} className={styles.accordion_item}>
              <div
                className={styles.accordion_header}
                role="button"
                tabIndex={0}
                onClick={() => toggleAccordion(index)}
                onKeyDown={(e) =>
                  (e.key === 'Enter' || e.key === ' ') && toggleAccordion(index)
                }
              >
                <div className="flex space_between">
                  <div className="flex">
                    <p className={styles.title}>{event.time}</p>
                    <p className={styles.title}>{event.currency || 'Unknown'}</p>
                  </div>
                  <div className="flex">
                    <p>{event.event || 'No Title'}</p>
                  </div>
                </div>
              </div>

              {expandedIndex === index && (
                <div className={styles.accordion_content}>
                  <p>
                    <strong>Impact:</strong> {event.impact}
                  </p>
                  <p>
                    <strong>Time:</strong> {event.time}
                  </p>
                  <p>
                    <strong>Actual:</strong> {event.actual}
                  </p>
                  <p>
                    <strong>Forecast:</strong> {event.forecast}
                  </p>
                  <p>
                    <strong>Previous:</strong> {event.previous}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default Calendar;
