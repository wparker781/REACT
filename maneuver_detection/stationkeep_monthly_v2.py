#!/usr/bin/env python3
"""
Simple Space-Track downloader + optional "one TLE per object" reducer.

- Edit SPACETRACK_USER, SPACETRACK_PASS
- Edit DATES (UTC, YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
- Run:  python pull_tles_simple.py
"""

import os
from datetime import datetime, timedelta
from io import StringIO
import pandas as pd
import requests
import time
# plot the SMA vs time for satcat 25544
import matplotlib.pyplot as plt
import numpy as np

# --- EDIT THESE ---
GET_DATA = False
CONVERT_TO_PICKLE = False
COMBINE_DATA = True
SPACETRACK_USER = os.getenv("SPACETRACK_USER") or "wparker@mit.edu"
SPACETRACK_PASS = os.getenv("SPACETRACK_PASS") or "AlpoAlpoAlpoAlpo1!"

# make DATES every month between start and end
start_date = '2000-01-01'
end_date = '2025-09-01'
start = datetime.strptime(start_date, '%Y-%m-%d')
end = datetime.strptime(end_date, '%Y-%m-%d')
DATES = []
current = start
while current <= end:
    DATES.append(current.strftime('%Y-%m-%d'))
    if current.month == 12:
        current = current.replace(year=current.year + 1, month=1)
    else:
        current = current.replace(month=current.month + 1)

REDUCE_TO_ONE_PER_OBJECT = True  # set False to skip reduction
# ------------------

# --- new imports ---
import random
from collections import deque

# --- config your safety margins (below the hard caps) ---
MAX_PER_MIN = 25   # stay below 30/min
MAX_PER_HOUR = 250 # stay below 300/hour

def throttle_factory(max_per_min=MAX_PER_MIN, max_per_hour=MAX_PER_HOUR):
    """Return a throttle() func that enforces <max/min> & <max/hour> with jitter."""
    calls_min = deque()
    calls_hour = deque()
    def throttle():
        now = time.time()
        # trim old timestamps
        while calls_min and now - calls_min[0] >= 60: calls_min.popleft()
        while calls_hour and now - calls_hour[0] >= 3600: calls_hour.popleft()
        # if at limit, sleep until we’re safe
        wait = 0.0
        if len(calls_min) >= max_per_min:
            wait = max(wait, 60 - (now - calls_min[0]))
        if len(calls_hour) >= max_per_hour:
            wait = max(wait, 3600 - (now - calls_hour[0]))
        if wait > 0:
            time.sleep(wait + 0.1)
        # small jitter to avoid thundering herd
        time.sleep(0.2 + random.random()*0.4)
        t = time.time()
        calls_min.append(t); calls_hour.append(t)
    return throttle

def safe_get(session: requests.Session, url: str, max_retries: int = 6):
    """GET with throttle + 429/503 backoff + Retry-After support."""
    throttle = safe_get._throttle
    backoff = 2.0
    for attempt in range(max_retries):
        throttle()
        r = session.get(url, timeout=120)
        # explicit rate-limit handling
        if r.status_code in (429, 503):
            ra = r.headers.get("Retry-After")
            sleep_s = float(ra) if ra and ra.isdigit() else backoff
            time.sleep(sleep_s)
            backoff = min(backoff * 2.0, 60.0)
            continue
        # other HTTP errors
        if r.status_code >= 400:
            r.raise_for_status()
        # sometimes Space-Track returns HTML error pages; detect and retry
        ct = r.headers.get("Content-Type","")
        if "text/html" in ct and "Space-Track" in r.text:
            time.sleep(backoff); backoff = min(backoff*2.0, 60.0)
            continue
        return r
    raise RuntimeError(f"Giving up after {max_retries} retries for {url}")

# bind a throttle instance for this run
safe_get._throttle = throttle_factory()

def login(session: requests.Session) -> None:
    session.headers.update({
        "User-Agent": "will-parker-tle-dl/1.0 (+contact: your_email@example.com)"
    })
    r = session.post(
        "https://www.space-track.org/ajaxauth/login",
        data={"identity": SPACETRACK_USER, "password": SPACETRACK_PASS},
        timeout=30,
    )
    r.raise_for_status()

def parse_dt(s: str) -> datetime:
    # allows "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS"
    dt = datetime.fromisoformat(s)
    # treat naive as UTC
    return dt

def one_day_window(start: datetime):
    return start, start + timedelta(days=1)

def login(session: requests.Session) -> None:
    r = session.post(
        "https://www.space-track.org/ajaxauth/login",
        data={"identity": SPACETRACK_USER, "password": SPACETRACK_PASS},
        timeout=30,
    )
    r.raise_for_status()

def build_url(start_iso: str, end_iso: str) -> str:
    # Use your exact encoding: EPOCH/%3Estart%2C%3Cend
    # Note: if you include times, make sure they’re ISO like "2024-01-01 06:00:00"
    start_enc = start_iso.replace(" ", "%20").replace(":", "%3A")
    end_enc = end_iso.replace(" ", "%20").replace(":", "%3A")
    return (
        "https://www.space-track.org/basicspacedata/query/"
        f"class/tle/EPOCH/%3E{start_enc}%2C%3C{end_enc}/"
        "orderby/NORAD_CAT_ID%20asc/format/csv/emptyresult/show"
    )
def reduce_one_per_object(csv_text: str) -> str:
    # for each row in the csv, get the NORAD_CAT_ID, and if we've seen it already, mark the row for deletion
    i = 0
    seen = []
    to_delete = []
    lines = csv_text.splitlines()
    for line in lines:
        if i == 0:
            # header line
            i += 1
            continue
        norad = (line.split(',')[2])
        if norad in seen:
            to_delete.append(i)
        else:
            seen.append(norad)
        i += 1

    # Remove duplicates
    for i in sorted(to_delete, reverse=True):
        lines.pop(i)

    sma = []
    satcat = []
    obj_type = []
    obj_name = []
    for i in range(len(lines)):
        if i != 0:
            sma.append(float(lines[i].split(',')[27][1:-1]))
            satcat.append(int(lines[i].split(',')[2][1:-1]))
            obj_type.append(lines[i].split(',')[4][1:-1])
            obj_name.append(lines[i].split(',')[3][1:-1])

    return satcat, sma, obj_type, obj_name

# def reduce_one_per_object(csv_text: str) -> str:
#     """
#     Return classic 3-line TLE text with exactly one (latest-by-EPOCH)
#     TLE per NORAD_CAT_ID from a Space-Track `class=tle&format=csv` response.
#     If no usable rows, return "".
#     """
#     import io, csv
#     from datetime import datetime

#     f = io.StringIO(csv_text)
#     reader = csv.DictReader(f)
#     if not reader.fieldnames:
#         return ""

#     # Normalize field names (strip BOM, spaces; make a lookup by UPPER name)
#     fieldnames = [ (fn or "").lstrip("\ufeff").strip() for fn in reader.fieldnames ]
#     uppermap = {fn.upper(): fn for fn in fieldnames}

#     required = ["NORAD_CAT_ID", "EPOCH", "TLE_LINE1", "TLE_LINE2"]
#     if any(k not in uppermap for k in required):
#         # Probably an empty CSV (just headers) or unexpected schema
#         return ""

#     def G(row, key):
#         # helper to get a column by canonical name
#         return row.get(uppermap[key], "").strip()

#     best = {}  # norad -> (epoch_dt, l1, l2)

#     for row in reader:
#         norad_s = G(row, "NORAD_CAT_ID")
#         epoch_s = G(row, "EPOCH")
#         l1 = G(row, "TLE_LINE1")
#         l2 = G(row, "TLE_LINE2")
#         if not norad_s or not epoch_s or not l1 or not l2:
#             continue
#         # NORAD as int (be lenient about weird strings)
#         try:
#             norad = int(float(norad_s))
#         except Exception:
#             continue
#         # Parse EPOCH; treat as UTC; accept ‘YYYY-MM-DD HH:MM:SS[.us]’ or with ‘T’
#         ep_str = epoch_s.replace("T", " ").replace("Z", "")
#         try:
#             # fromisoformat handles "YYYY-MM-DD HH:MM:SS[.ffffff]"
#             ep = datetime.fromisoformat(ep_str)
#         except Exception:
#             continue

#         prev = best.get(norad)
#         if prev is None or ep > prev[0]:
#             best[norad] = (ep, l1.rstrip(), l2.rstrip())

#     if not best:
#         return ""

#     # Emit classic 3-line TLE, sorted by NORAD
#     out_lines = []
#     for norad in sorted(best):
#         _, l1, l2 = best[norad]
#         out_lines.append(f"0 {norad}\n{l1}\n{l2}\n")
#     return "".join(out_lines)

def main():
    # os.makedirs("csv", exist_ok=True)
    # os.makedirs("out", exist_ok=True)

    # check to see if desired csv files already exist
    existing_files = os.listdir("csv")
    existing_dates = [f.split('_')[1].split('.')[0] for f in existing_files if f.startswith('tle_') and f.endswith('.csv')]
    dates_to_fetch = [d for d in DATES if d not in existing_dates]

    if GET_DATA == True:
        if not dates_to_fetch:
            print("All desired CSV files already exist. Exiting.")
        else:
            print(f"Fetching {len(dates_to_fetch)} new CSV files...")

            with requests.Session() as s:
                login(s)
                for d in DATES:
                    start = parse_dt(d)
                    end = one_day_window(start)[1]
                    start_iso = start.strftime("%Y-%m-%d %H:%M:%S")
                    end_iso = end.strftime("%Y-%m-%d %H:%M:%S")
                    url = build_url(start_iso, end_iso)

                    print(f"Fetching {start_iso} -> {end_iso}")
                    # r = s.get(url, timeout=120)
                    r = safe_get(s, url)
                    r.raise_for_status()
                    csv_text = r.text

                    # Save raw CSV
                    csv_path = f"csv/tle_{start.strftime('%Y-%m-%d')}.csv"
                    with open(csv_path, "w", encoding="utf-8") as f:
                        f.write(csv_text)
                    print(f"  wrote CSV: {csv_path}")

                    # check to see if the csv has any data rows (beyond header)
                    lines = csv_text.splitlines()
                    if len(lines) <= 1:
                        time.sleep(15)
                        print(f"  no data rows in CSV, waiting 15s before next fetch")
                    time.sleep(3)

    if CONVERT_TO_PICKLE == True:
        print(f"Converting CSV files to pickles...")
        for d in DATES:
            start = parse_dt(d)
            csv_path = f"csv/tle_{start.strftime('%Y-%m-%d')}.csv"
            with open(csv_path, "r", encoding="utf-8") as f:
                csv_text = f.read()
            if REDUCE_TO_ONE_PER_OBJECT:
                satcat, sma, obj_type, obj_name = reduce_one_per_object(csv_text)
                # save satcat and sma to a pkl file
                df = pd.DataFrame({'NORAD_CAT_ID': satcat, 'SEMIMAJOR_AXIS_KM': sma, 'OBJECT_TYPE': obj_type, 'OBJECT_NAME': obj_name})
                df.to_pickle(f'out/satcat_sma_{start.strftime("%Y-%m-%d")}.pkl')
                print(f"  wrote satcat/semimajor axis pkl: out/satcat_sma_{start.strftime('%Y-%m-%d')}.pkl")

    if COMBINE_DATA == True:
        print(f"Combining all satcat pickles into one...")
        # read the pickle files and combine into one dataframe with the norad as the key and the dates in teh filenames as the timestamps
        # pkl_files = os.listdir("out")
        # pkl_files = [f for f in pkl_files if f.startswith('satcat')]
        
        import re
        pkl_files = [
            f for f in os.listdir("out")
            if re.match(r"^satcat_sma_\d{4}-\d{2}-\d{2}\.pkl$", f)
        ]

        dfs = []
        for f in pkl_files:
            df = pd.read_pickle(f'out/{f}')
            df['TIMESTAMP'] = f.split('_')[-1].split('.')[0]
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)

        satcat_25544 = combined[combined['NORAD_CAT_ID'] == 25544]
        satcat_25544 = satcat_25544.sort_values(by='TIMESTAMP')
        bad = satcat_25544[~satcat_25544['TIMESTAMP'].str.match(r'^\d{4}-\d{2}-\d{2}')].head()
        print("Bad TIMESTAMP rows:\n", bad)
        satcat_25544['TIMESTAMP'] = pd.to_datetime(satcat_25544['TIMESTAMP'])
        plt.plot(satcat_25544['TIMESTAMP'], satcat_25544['SEMIMAJOR_AXIS_KM'], marker='o')
        plt.xlabel('Date')
        plt.ylabel('Semimajor Axis (km)')
        plt.title('ISS (NORAD 25544) Semimajor Axis Over Time')
        plt.grid()
        plt.show()




        # for each satellite, difference the semimajor axis from the previous month
        combined = combined.sort_values(by=['NORAD_CAT_ID', 'TIMESTAMP'])
        combined['SMA_DIFF_KM'] = combined.groupby('NORAD_CAT_ID')['SEMIMAJOR_AXIS_KM'].diff()

        # compute statistics on the sma_diff_km
        stats = combined.groupby('NORAD_CAT_ID')['SMA_DIFF_KM'].agg(['mean', 'std', 'count']).reset_index()
        stats = stats.rename(columns={'mean': 'SMA_DIFF_MEAN_KM', 'std': 'SMA_DIFF_STDDEV_KM', 'count': 'SMA_DIFF_OBS_COUNT'})

        # normalize the sma_diff_km by the mean and stdev for each satellite
        combined = combined.merge(stats[['NORAD_CAT_ID', 'SMA_DIFF_MEAN_KM', 'SMA_DIFF_STDDEV_KM']], on='NORAD_CAT_ID', how='left')
        combined['SMA_DIFF_NORM'] = (combined['SMA_DIFF_KM'] - combined['SMA_DIFF_MEAN_KM']) / combined['SMA_DIFF_STDDEV_KM']

        # plot sma_diff_norm wrt time for satcat 25544 and 11
        sma_diff_norm_25544 = combined[combined['NORAD_CAT_ID'] == 25544]
        sma_diff_norm_25544 = sma_diff_norm_25544.sort_values(by='TIMESTAMP')
        sma_diff_norm_25544['TIMESTAMP'] = pd.to_datetime(sma_diff_norm_25544['TIMESTAMP'])
        plt.plot(sma_diff_norm_25544['TIMESTAMP'], sma_diff_norm_25544['SMA_DIFF_NORM'], marker='o')
        plt.xlabel('Date')
        plt.ylabel('Normalized SMA Difference')
        plt.title('ISS (NORAD 25544) Normalized SMA Difference Over Time')
        plt.grid()
        plt.show()

        # repeat above for all debris objects (obj_type = 'DEBRIS') and plot each one
        debris = combined[combined['OBJECT_TYPE'] == 'DEBRIS']
        # debris_satcats = debris['NORAD_CAT_ID'].unique()
        # for satcat in debris_satcats:
        #     debris_satcat = debris[debris['NORAD_CAT_ID'] == satcat]
        #     debris_satcat = debris_satcat.sort_values(by='TIMESTAMP')
        #     debris_satcat['TIMESTAMP'] = pd.to_datetime(debris_satcat['TIMESTAMP'])
        #     plt.plot(debris_satcat['TIMESTAMP'], debris_satcat['SMA_DIFF_NORM'], marker='o', label=str(satcat))
        #     plt.xlabel('Date')
        #     plt.ylabel('Normalized SMA Difference')
        #     plt.title(f'Debris NORAD {satcat} Normalized SMA Difference Over Time')
        #     plt.grid()
        #     plt.legend()
        #     plt.show()

        # repeat above, but only include timestamps where the object has sma > 6378.15 + 400 km. 
        # debris_high = debris[debris['SEMIMAJOR_AXIS_KM'] > 6378.15 + 400]
        # repeat line above, but also require that sma < 6378.15 + 1000 km
        debris_high = debris[(debris['SEMIMAJOR_AXIS_KM'] > 6378.15 + 400) & (debris['SEMIMAJOR_AXIS_KM'] < 6378.15 + 800)]
        debris_satcats = debris_high['NORAD_CAT_ID'].unique()
        plt.figure()
        j = 0
        for satcat in debris_satcats:
            if j % 100 == 0:
                debris_satcat = debris_high[debris_high['NORAD_CAT_ID'] == satcat]
                debris_satcat = debris_satcat.sort_values(by='TIMESTAMP')
                debris_satcat['TIMESTAMP'] = pd.to_datetime(debris_satcat['TIMESTAMP'])
                plt.plot(debris_satcat['TIMESTAMP'], debris_satcat['SMA_DIFF_NORM'], label=str(satcat), color='gray', alpha=0.3, linewidth=0.5)
                
            j += 1

        # record the SMA_diff_norm mean and stdev for each timestep across all debris objects where sma > 400 and have a value at that timestamp
        debris_stats = debris_high.groupby('TIMESTAMP')['SMA_DIFF_NORM'].agg(['mean', 'std', 'count']).reset_index()
        debris_stats = debris_stats.rename(columns={'mean': 'DEBRIS_SMA_DIFF_NORM_MEAN', 'std': 'DEBRIS_SMA_DIFF_NORM_STDDEV', 'count': 'DEBRIS_SMA_DIFF_NORM_OBS_COUNT'})
        debris_stats['TIMESTAMP'] = pd.to_datetime(debris_stats['TIMESTAMP'])
        plt.plot(debris_stats['TIMESTAMP'], debris_stats['DEBRIS_SMA_DIFF_NORM_MEAN'], label='Mean', color='r', linewidth=2)
        plt.fill_between(debris_stats['TIMESTAMP'], 
                         debris_stats['DEBRIS_SMA_DIFF_NORM_MEAN'] - 3*debris_stats['DEBRIS_SMA_DIFF_NORM_STDDEV'], 
                         debris_stats['DEBRIS_SMA_DIFF_NORM_MEAN'] + 3*debris_stats['DEBRIS_SMA_DIFF_NORM_STDDEV'], 
                         color='r', alpha=0.4)
        plt.plot(sma_diff_norm_25544['TIMESTAMP'], sma_diff_norm_25544['SMA_DIFF_NORM'], color = 'k', alpha=0.5, linewidth=1)
        plt.xlabel('Date')
        plt.ylabel('Normalized SMA Difference')
        plt.title(f'Debris NORAD {satcat} (SMA > 400km) Normalized SMA Difference Over Time')
        plt.grid()
        # plt.legend()
        plt.show()


        # for each satellite, compute the mean and stddev of the semimajor axis
        stats = combined.groupby('NORAD_CAT_ID')['SEMIMAJOR_AXIS_KM'].agg(['mean', 'std', 'count']).reset_index()
        stats = stats.rename(columns={'mean': 'SMA_MEAN_KM', 'std': 'SMA_STDDEV_KM', 'count': 'OBS_COUNT'})



        combined.to_pickle('out/satcat_combined.pkl')
        print(f"  wrote combined satcat pkl: out/satcat_combined.pkl")

if __name__ == "__main__":
    main()
