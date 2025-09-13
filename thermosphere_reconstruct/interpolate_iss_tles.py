# load iss_combined_tles and interpolate semi-major axis so that we have a timestamped hourly SMA for ISS

#!/usr/bin/env python3
"""
Load a combined TLE text file and produce an hourly semi-major axis (SMA) time series.

- Expects a file that contains many TLEs back-to-back.
- Lines that start with '1 ' are line 1; lines that start with '2 ' are line 2.
- Epoch is read from line 1 (cols 19–32, YYDDD.DDDDDDDD).
- Mean motion (rev/day) is read from line 2 (cols 54–63 in 1-based indexing).
- SMA is computed from mean motion via Kepler's third law.

Output: CSV with columns timestamp_utc,sma_km (hourly, interpolated).
"""

import argparse
from datetime import datetime, timedelta, timezone
import math
import pandas as pd

MU_EARTH_KM3_S2 = 398600.4418  # km^3/s^2

def parse_epoch_from_line1(line1: str) -> datetime:
    # TLE epoch at columns 19-32 (0-based slice 18:32) -> YYDDD.DDDDDDDD
    s = line1[18:32]
    yy = int(s[0:2])
    year = 2000 + yy if yy < 57 else 1900 + yy  # standard TLE year rule
    doy = float(s[2:])  # day-of-year with fractional days
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1.0)

def parse_mean_motion_from_line2(line2: str) -> float:
    # Mean motion rev/day at columns 53-63 (0-based 52:63). Fallback to last token if needed.
    mm_str = line2[52:63].strip()
    if not mm_str:
        mm_str = line2.split()[-1]
    return float(mm_str)

def mean_motion_to_sma_km(n_rev_per_day: float) -> float:
    # Convert rev/day -> rad/s
    n_rad_s = n_rev_per_day * 2.0 * math.pi / 86400.0
    # Kepler: n = sqrt(mu/a^3) => a = (mu / n^2)^(1/3)
    return (MU_EARTH_KM3_S2 / (n_rad_s ** 2)) ** (1.0 / 3.0)

def read_tles_and_compute_sma(path: str):
    epochs = []
    smas = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln.rstrip('\n') for ln in f]
    i = 0
    while i < len(lines):
        l = lines[i].lstrip()
        if l.startswith('1 ') and i + 1 < len(lines):
            l1 = lines[i]
            l2 = lines[i + 1].lstrip()
            if l2.startswith('2 '):
                try:
                    epoch = parse_epoch_from_line1(l1)
                    n_rev_day = parse_mean_motion_from_line2(l2)
                    sma_km = mean_motion_to_sma_km(n_rev_day)
                    # Filter obvious junk (ISS ~ 6770–6800 km SMA)
                    if 6400.0 < sma_km < 7200.0:
                        epochs.append(epoch)
                        smas.append(sma_km)
                except Exception:
                    # Skip malformed pairs silently
                    pass
                i += 2
                continue
        i += 1
    return pd.DataFrame({'timestamp_utc': epochs, 'sma_km': smas})

def make_hourly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values('timestamp_utc').drop_duplicates('timestamp_utc')
    df = df.set_index('timestamp_utc')
    # Build hourly grid from first to last observation
    idx_hourly = pd.date_range(df.index.min(), df.index.max(), freq='H', tz=timezone.utc)
    df_hourly = df.reindex(idx_hourly)
    # Time-aware interpolation between TLE epochs
    df_hourly['sma_km'] = df_hourly['sma_km'].interpolate(method='time')
    # Optionally fill tiny leading/trailing gaps if present
    df_hourly['sma_km'] = df_hourly['sma_km'].ffill().bfill()
    df_hourly.index.name = 'timestamp_utc'
    return df_hourly.reset_index()

def main():
    p = argparse.ArgumentParser(description="Interpolate ISS SMA from combined TLEs to hourly cadence.")
    p.add_argument('--input', '-i', default='iss_combined_tles.txt',
                   help="Path to combined TLE file (default: iss_combined_tles.txt)")
    p.add_argument('--output', '-o', default='iss_sma_hourly.csv',
                   help="Output CSV path (default: iss_sma_hourly.csv)")
    args = p.parse_args()

    input_path = 'combined_iss_tles.txt'

    df_obs = read_tles_and_compute_sma(input_path)
    if df_obs.empty:
        raise SystemExit(f"No valid TLEs found in {input_path}")

    df_hourly = make_hourly(df_obs)
    df_hourly.to_csv(args.output, index=False)
    print(f"Parsed {len(df_obs)} TLE epochs; wrote {len(df_hourly)} hourly samples to {args.output}")

if __name__ == '__main__':
    main()
