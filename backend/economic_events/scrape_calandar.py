from datetime import datetime, date as DateType
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from fake_useragent import UserAgent
from utils.helpers import red, reset
from typing import List, Dict, Any, Union


def get_user_agent() -> str:
    return UserAgent().random


def scrape_forexfactory_day(date: Union[str, DateType]) -> List[Dict[str, Any]]:
    if isinstance(date, str):
        try:
            date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            try:
                date = datetime.strptime(date, "%Y/%m/%d").date()
            except ValueError:
                print(
                    red
                    + f"scrape_forexfactory_day: Could not convert '{date}' to date object"
                    + reset
                )
                raise ValueError(f"Could not convert string '{date}' to date object")

    date_str = date.strftime("%b%d.%Y")
    url = f"https://www.forexfactory.com/calendar?day={date_str}"

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(f"user-agent={get_user_agent()}")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(service=Service(), options=chrome_options)
    events: List[Dict[str, Any]] = []

    try:
        driver.get(url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table.calendar__table"))
        )
        time.sleep(5)

        rows = driver.find_elements(By.CSS_SELECTOR, "tr.calendar__row")
        last_time = "00:01"

        for row in rows:
            try:
                time_text = row.find_element(
                    By.CSS_SELECTOR, ".calendar__time"
                ).text.strip()
                if time_text == "All Day":
                    time_text = "00:01"
                if not time_text:
                    time_text = last_time
                else:
                    last_time = time_text

                try:
                    time_24 = datetime.strptime(time_text, "%I:%M%p").strftime("%H:%M")
                except ValueError:
                    time_24 = "00:01"

                event = {
                    "date": date.strftime("%Y-%m-%d"),
                    "time": time_24,
                    "currency": row.find_element(
                        By.CSS_SELECTOR, ".calendar__currency"
                    ).text.strip(),
                    "event": row.find_element(
                        By.CSS_SELECTOR, ".calendar__event"
                    ).text.strip(),
                    "impact": row.find_element(
                        By.CSS_SELECTOR, ".calendar__impact .icon"
                    ).get_attribute("title"),
                    "actual": row.find_element(
                        By.CSS_SELECTOR, ".calendar__actual"
                    ).text.strip(),
                    "forecast": row.find_element(
                        By.CSS_SELECTOR, ".calendar__forecast"
                    ).text.strip(),
                    "previous": row.find_element(
                        By.CSS_SELECTOR, ".calendar__previous"
                    ).text.strip(),
                }
                events.append(event)
            except Exception:
                continue
    except Exception as e:
        print(red + f"Error scraping {date_str}: {e}" + reset)
    finally:
        driver.quit()

    return events


if __name__ == "__main__":
    from datetime import date

    print(scrape_forexfactory_day(date.today()))
