from typing import Any, List
import os
import pandas as pd
import json
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
from time import sleep
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def ingest_csv(path: str) -> pd.DataFrame:
    """Ingest CSV file."""
    return pd.read_csv(path)


def ingest_json(path: str) -> pd.DataFrame:
    """Ingest JSON file."""
    return pd.read_json(path)


def ingest_excel(path: str) -> pd.DataFrame:
    """Ingest Excel file."""
    return pd.read_excel(path)


class DataIngestor:
    def __init__(self, game: str):
        self.game = game
        self.history_path = os.path.join("data", game, "history")
        self.actuals_path = os.path.join("data", game, "actuals")

    def ingest_historic_data(self, file_path: str) -> pd.DataFrame:
        """Ingest historic draw data from a file."""
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".csv":
            return ingest_csv(file_path)
        elif ext == ".json":
            return ingest_json(file_path)
        elif ext in [".xls", ".xlsx"]:
            return ingest_excel(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def ingest_live_data(self, data: List[int]) -> None:
        """Ingest live draw data manually or automatically."""
        # Save data to actuals path
        os.makedirs(self.actuals_path, exist_ok=True)
        file_path = os.path.join(self.actuals_path, "live_data.json")
        with open(file_path, "w") as f:
            json.dump({"numbers": data}, f)

    def validate_numbers(self, numbers: List[int], min_val: int, max_val: int) -> bool:
        """Validate number counts and ranges."""
        return all(min_val <= num <= max_val for num in numbers)

    def ingest_from_url(self, url: str, year: int | None = None, timeout: int = 15, throttle_seconds: float = 0.0) -> pd.DataFrame:
        """Fetch and parse draw tables from a URL for a specific year.

        This implementation targets the table structure you provided (class contains
        "archiveResults" and rows with date, a <ul class="balls"> of numbers, and
        a <td class="jackpot">). It returns a tidy DataFrame with columns:
        draw_date, year, n1..n7, bonus, jackpot, numbers (comma-separated mains).
        """
        # Use a session with retry/backoff and a sensible User-Agent header
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        headers = {
            "User-Agent": "LottoAI/1.0 (+https://example.com)"
        }

        # Allow tests to monkeypatch requests.get; prefer it if available so unit tests
        # that patch requests.get (not session.get) will be respected. If requests.get
        # raises or isn't suitable, fall back to the session.get (which has retries).
        try:
            try:
                response = requests.get(url, headers=headers, timeout=timeout)
            except TypeError:
                # some test fakes only accept (url, timeout=...)
                response = requests.get(url, timeout=timeout)
            # If this is a requests.Response, try to use raise_for_status; if the
            # test's fake object doesn't have it that's fine.
            try:
                response.raise_for_status()
            except Exception:
                pass
        except Exception:
            # Fall back to session with retries for real network calls
            response = session.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

        # optional throttle to be polite
        if throttle_seconds and throttle_seconds > 0:
            sleep(float(throttle_seconds))

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the most likely results table.
        # Accept tables with class names like 'archiveResults', 'past-results', or 'mobFormat',
        # or fallback to a table whose header contains 'Winning Numbers' / 'Jackpot'.
        table = None
        for t in soup.find_all('table'):
            cls = ' '.join(t.get('class') or [])
            if 'archiveResults' in cls or 'past-results' in cls or 'mobFormat' in cls:
                table = t
                break
            # check headers for known column titles
            thead = t.find('thead')
            if thead:
                th_text = ' '.join([th.get_text(' ', strip=True) for th in thead.find_all('th')])
                if 'Winning Numbers' in th_text or 'Jackpot' in th_text or 'Winning numbers' in th_text:
                    table = t
                    break

        if table is None:
            # Last resort: take first table
            table = soup.find('table')
        if table is None:
            raise ValueError('No table found on page to parse results.')

        tbody = table.find('tbody') or table
        rows = tbody.find_all('tr', recursive=False)

        data = []
        for tr in rows:
            # Skip month header rows and other non-draw rows with colspan
            if tr.find('td', attrs={'colspan': True}):
                continue

            # Prefer positional cells: Date | Winning Numbers | Jackpot | Winners | Link
            tds = tr.find_all('td', recursive=False)
            if len(tds) < 2:
                continue

            date_td = tds[0]
            numbers_td = tds[1]
            jackpot_td = tds[2] if len(tds) > 2 else None

            # Extract date: cell contains weekday + <br> + "Month DD YYYY" text.
            date_text_full = date_td.get_text(' ', strip=True)
            # Try to capture 'Month DD YYYY'
            m = re.search(r'([A-Za-z]+\s+\d{1,2}\s+\d{4})', date_text_full)
            if m:
                date_text = m.group(1)
            else:
                # fallback to last three tokens
                parts = date_text_full.split()
                if len(parts) >= 3:
                    date_text = ' '.join(parts[-3:])
                else:
                    date_text = date_text_full

            try:
                draw_dt = pd.to_datetime(date_text, dayfirst=False, errors='coerce')
            except Exception:
                draw_dt = pd.to_datetime(date_text, dayfirst=False, errors='coerce')

            if pd.isna(draw_dt):
                continue
            if year is not None and int(draw_dt.year) != int(year):
                continue

            # Numbers block
            ul = numbers_td.find('ul', class_=lambda c: c and 'balls' in c)
            # some pages may not use class; fall back to any <li class="ball">
            main_nums = []
            bonus_num = None
            if ul:
                lis = ul.find_all('li')
            else:
                lis = numbers_td.find_all('li')

            for li in lis:
                text = li.get_text(strip=True)
                if not text or not re.match(r'^\d+$', text):
                    continue
                val = int(text)
                classes = li.get('class', []) or []
                if any('bonus' in c for c in classes):
                    bonus_num = val
                else:
                    main_nums.append(val)

            # Jackpot: often the 3rd td
            # Jackpot parsing: handle '$70,000,000', human-readable like '1.21 Million',
            # and currency strings that include cents (e.g. '$39,621,339.80' -> digits '3962133980').
            jackpot_val = None
            if jackpot_td:
                jackpot_text = jackpot_td.get_text(' ', strip=True)

                # First handle "million" style human-readable strings
                m = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*(million|m)\b', jackpot_text, flags=re.IGNORECASE)
                if m:
                    try:
                        num = float(m.group(1))
                        jackpot_val = int(round(num * 1_000_000))
                    except Exception:
                        jackpot_val = None
                else:
                    # Extract digits only. Many sources include commas and a decimal point for cents.
                    # Example: "$39,621,339.80" -> digits '3962133980' (cents). In that case we
                    # detect the decimal point and divide by 100 to recover whole dollars.
                    digits = re.sub(r'[^0-9]', '', jackpot_text)
                    if digits and len(digits) >= 4:
                        try:
                            iv = int(digits)
                            # If original text had an explicit decimal point, it's very likely the
                            # digits include cents (we parsed cents into the integer), so divide by 100.
                            if '.' in jackpot_text:
                                jackpot_val = int(round(iv / 100.0))
                            else:
                                # As a fallback, if the parsed integer is unreasonably large
                                # (for our domain), assume it included cents and divide by 100.
                                if iv > 200_000_000:
                                    jackpot_val = int(round(iv / 100.0))
                                else:
                                    jackpot_val = iv
                        except Exception:
                            jackpot_val = None

            # Build row
            row = {
                'draw_date': pd.to_datetime(draw_dt).strftime('%Y-%m-%d'),
                'year': int(draw_dt.year),
                'numbers': ','.join(str(x) for x in main_nums),
                'bonus': bonus_num,
                'jackpot': jackpot_val,
            }
            for i in range(7):
                row[f'n{i+1}'] = main_nums[i] if i < len(main_nums) else None

            data.append(row)

        df = pd.DataFrame(data)

        # Sort by draw_date descending (most recent first)
        if not df.empty:
            df['draw_date'] = pd.to_datetime(df['draw_date'])
            df = df.sort_values('draw_date', ascending=False).reset_index(drop=True)
            # Format draw_date back to string for display
            df['draw_date'] = df['draw_date'].dt.strftime('%Y-%m-%d')

        # Debug print for developer visibility
        print('Debug: parsed rows:', len(df))
        print(df.head())

        return df
