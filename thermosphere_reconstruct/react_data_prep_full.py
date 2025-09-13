# Full REACT reconstruction pipeline
# William Parker, 2023

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import datetime as dt
import pandas as pd
from datetime import datetime
from spacetrack import SpaceTrackClient
import spacetrack.operators as op
import datetime as dt
import time
import csv
from pathlib import Path
import os
import math
from datetime import timedelta
from collections import defaultdict
from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs72
from tqdm import tqdm
import numpy as np
import datetime as dt
from sgp4.api import Satrec
from sgp4.conveniences import jday_datetime
import numpy as np
from datetime import datetime, timedelta
from pyproj import Transformer
import matplotlib.pyplot as plt
from sgp4.api import jday  # add this to your imports
from astropy.time import Time
from astropy.coordinates import TEME, ITRS, CartesianRepresentation
from astropy.coordinates import EarthLocation
import astropy.units as u
import pickle as pkl
from datetime import datetime, timedelta
import numpy as np
from cysgp4 import PyTle, PyDateTime, propagate_many
import requests

# Start by identifying the dates of interest
run_name = 'Gannon'
start_date = dt.datetime(2024, 5, 9)
end_date = dt.datetime(2024, 5, 15)
t_m = 10 # minutes between interpolated points
dimensions = 1

# Space-Track credentials
# ask user to input their Space-Track credentials
st_email = input("Enter your Space-Track email: ")
st_pass = input("Enter your Space-Track password: ")

def main():
    # STEP 1: GET RELEVANT OBJECT NORADS
    # Identify the NORADs that are debris and approximately circular for a target time
    t1 = time.time()
    print(f"Extracting debris NORADs for {start_date.strftime('%Y-%m-%d')}...")
    get_debris_norads(start_date, run_name, st_email, st_pass)
    print(f"NORAD extraction complete.")
    print(time.time() - t1)

    # STEP 2: PULL TLEs FOR SATCATS DURING THE WINDOW OF INTEREST
    print(f"Pulling TLEs for debris NORADs from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    pull_tles_leo_deb(start_date, end_date, run_name, st_email, st_pass)
    print(f"TLE extraction complete.")
    print(time.time() - t1)

    # STEP 3: COMPUTE d_s FOR EACH SEQUENTIAL TLE SET
    print(f"Computing ds from TLEs for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} with {t_m} minute intervals...")
    ds_from_tles(start_date, end_date, run_name, t_m)
    print(f"Density estimates saved for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")
    print(time.time() - t1)

    # if dimensions == 1:
        # print("Running 1D reconstruction...")
        # from recon_1d_expo import main as recon_1d_expo_main
        # recon_1d_expo_main()

def ds_from_tles(start_date, end_date, run_name, t_m):

    # check to see if file already exists
    op_file = 'data/' + run_name + '_' + start_date.strftime('%Y%m%d') + '_' + str(t_m) + 'm_ds.pkl'

    if os.path.exists(op_file):
        print(f"File {op_file} already exists. Skipping ds extraction.")
        return

    # Set up ECEF to geodetic transformer (WGS84)
    ecef2geo = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)

    mu = 3.986004418e5  # km^3/s^2
    RE = 6378.135

    tle_dir = 'data/' + run_name + '_' + start_date.strftime('%Y%m%d') + '_tles/'

    grouped_tles = defaultdict(list)

    satcat_list = []
    start_time_list = []
    end_time_list = []
    dn_list = []
    lat_list = []
    lon_list = []
    alt_list = []
    time_list = []

    # Step 1: Load and group TLEs by NORAD ID
    for fname in tqdm(os.listdir(tle_dir)):
        if not fname.endswith('.txt'):
            continue

        with open(os.path.join(tle_dir, fname), 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        for i in range(0, len(lines) - 1, 2):
            l1, l2 = lines[i], lines[i+1]
            try:
                norad_id = int(l1[2:7])
                satrec = twoline2rv(l1, l2, wgs72)
                grouped_tles[norad_id].append((satrec, l1, l2))
            except:
                continue
    results = []

    # make tvec start at the minimum epoch and end at the maximum epoch across all the tles
    min_epoch = start_date
    max_epoch = end_date

    # use timedelta = 20 minutes to create tvec
    tvec = np.arange(min_epoch, max_epoch, timedelta(minutes=t_m))

    # convert tvec to datetime objects
    tvec_dt = np.array([dt.datetime.fromisoformat(str(t)) for t in tvec])

    # Step 2: For each satellite, sort by epoch and compute densities
    a_lst = []
    for sat_id, tles in tqdm(grouped_tles.items(), desc="Processing satellites"):
        # sort oldest to newest
        tles = sorted(tles, key=lambda x: x[0].epoch)
        for i in range(len(tles) - 1):
            sat1, l1_1, l1_2 = tles[i]
            sat2, l2_1, _ = tles[i+1]

            t1 = sat1.epoch
            t2 = sat2.epoch
            delta_t = (t2 - t1).total_seconds()
            if delta_t <= 0:
                continue

            # find the indices of tvec that are between t1 and t2
            tvec_iter_idx = np.where((tvec >= t1) & (tvec <= t2))[0]
            # tvec_iter = tvec[(tvec >= t1) & (tvec <= t2)]
            tvec_dt_iter = tvec_dt[tvec_iter_idx]

            a1 = sat1.a * RE
            a2 = sat2.a * RE

            da = a2 - a1
            dadt = da / delta_t
            a_avg = 0.5 * (a1 + a2)

            d_n = dadt/(-np.sqrt(mu*a_avg))


            # compute lat, lon, alt for each timestep 
            # lats, lons, alts = propagate_tle_to_latlonalt(l1_1, l1_2, tvec_dt_iter)
            if len(tvec_iter_idx) > 0 and d_n > 0:
                lats, lons, alts = prop_tle_cysgp4(l1_1, l1_2, tvec_dt_iter)

                # record the satcat, start time, end time, d_n, and the lat, lon, alt, times in a dictionary
                satcat_list.append(sat_id)
                start_time_list.append(t1)
                end_time_list.append(t2)
                dn_list.append(d_n)
                lat_list.append(lats)
                lon_list.append(lons)
                alt_list.append(alts)
                time_list.append(tvec_dt_iter)
                a_lst.append(a_avg)

    alt = np.array(a_lst) - 6378.15


    with open(op_file, 'wb') as f:
        pkl.dump({
            'satcat': satcat_list,
            'start_time': start_time_list,
            'end_time': end_time_list,
            'ds': dn_list,
            'lat': lat_list,
            'lon': lon_list,
            'alt': alt_list,
            'time': time_list,
            'tvec': tvec_dt,
            'alt': alt
        }, f)

    print(f"✅ Saved {len(satcat_list)} satellites' TLE d_s estimates to {op_file}")

def prop_tle_cysgp4(tle_line1, tle_line2, times):

    tle = PyTle("ISS", tle_line1, tle_line2)

    # Convert times to Modified Julian Dates (MJD)
    mjds = np.array([[PyDateTime(t).mjd for t in times]])  # Shape: (1, time_steps)

    # Prepare TLEs array
    tles = np.array([[tle]])  # Shape: (1, 1)

    # Propagate satellite positions
    result = propagate_many(mjds.T, tles)

    # Extract geodetic positions
    geo = result["geo"]  # shape: (time_steps, 1, 3)
    lons = geo[:, 0, 0]
    lats = geo[:, 0, 1]
    alts = geo[:, 0, 2]

    return lats, lons, alts

def pull_tles_leo_deb(start_date, end_date, run_name, st_email, st_pass):

    # check to see if file already exists
    out_path = 'data/' + run_name + '_' + start_date.strftime('%Y%m%d') + '_tles/'
    if os.path.exists(out_path):
        print(f"Directory {out_path} already exists. Skipping TLE extraction.")
        return

    st = SpaceTrackClient(identity=st_email, password=st_pass)

    # Input list of SATCATs
    satcats = []
    fname = 'data/' + run_name + '_' + start_date.strftime('%Y%m%d') + '_debris_norads.txt'
    with open(fname, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            satcats.append(int(row[0]))

    drange = op.inclusive_range(start_date, end_date)

    # Output path
    Path(out_path).mkdir(parents=True, exist_ok=True)

    # Batch size
    batch_size = 100
    total_batches = len(satcats) // batch_size + int(len(satcats) % batch_size > 0)

    t0 = time.time()
    for i in range(0, len(satcats), batch_size):
        batch = satcats[i:i+batch_size]
        print(f"Querying batch {i//batch_size+1} of {total_batches} ({len(batch)} sats)...")
        
        try:
            tles = st.tle(
                norad_cat_id=','.join(map(str, batch)),
                epoch=drange,
                orderby='norad_cat_id,epoch desc',
                format='tle'
            )
            output_file = out_path + f'tle_batch_{i//batch_size+1}.txt'
            with open(output_file, 'w') as f:
                f.write(tles)
            print(f"Saved TLEs to {output_file}")
        except Exception as e:
            print(f"Failed on batch {i//batch_size+1}: {e}")

        time.sleep(1)  # polite delay to avoid rate limiting

    # Estimated time
    elapsed = (time.time() - t0) / 60
    print(f"\n All batches complete. Total time: {elapsed:.2f} minutes.")


def get_debris_norads(target_date, run_name, st_email, st_pass):
    start_date_str = target_date.strftime('%Y%m%d')
    fname =  'data/'+ run_name + '_' + start_date_str + '_debris_norads.txt'
    
    # if fname exists, print a message and end
    try:
        with open(fname, 'r') as f:
            print(f"File {fname} already exists. Skipping debris NORADs extraction.")
            return
    except FileNotFoundError:
        print(f"File {fname} does not exist. Proceeding with debris NORADs extraction.")
        
        # Load SATCAT CSV (downloaded from Space-Track)
        # if the satcat.csv file is out of date, download it again by logging into space-track.org
        # download through this link: https://www.space-track.org/basicspacedata/query/class/satcat
        # place the file in this folder to enable queries. 
        df = pd.read_csv('satcat.csv')

        # Parse dates and handle missing/invalid ones
        df['LAUNCH_DATE'] = pd.to_datetime(df['LAUNCH'], errors='coerce')
        df['DECAY'] = pd.to_datetime(df['DECAY'], errors='coerce')

        # Filter: launched before target date
        in_orbit = df[
            (df['LAUNCH_DATE'] < target_date) &
            (
                df['DECAY'].isna() |  # not decayed
                (df['DECAY'] > target_date)  # decayed after target date
            )
        ]

        # Filter only debris objects
        debris = in_orbit[in_orbit['OBJECT_TYPE'] == 'DEBRIS']

        # filter to objects with apogee < 1000 km
        debris = debris[debris['APOGEE'] < 1000]

        # Filter according to: 
        ecc_max = 0.05
        # filter to obly objects where (apogee-perigee)/(apogee+perigee) < ecc_max
        debris['ECCENTRICITY'] = (debris['APOGEE'] - debris['PERIGEE']) / (debris['APOGEE'] + debris['PERIGEE'])
        debris = debris[
            (debris['ECCENTRICITY'] < ecc_max)
        ]

        # Select relevant columns
        debris_list = debris[['NORAD_CAT_ID', 'OBJECT_NAME']]

        # create a list of the norad_cat_ids
        norad_cat_ids = debris_list['NORAD_CAT_ID'].tolist()

        # save norad_cat_ids to a txt file
        with open('data/'+ run_name + '_' + start_date_str + '_debris_norads.txt', 'w') as f:
            for norad_cat_id in norad_cat_ids:
                f.write(f"{norad_cat_id}\n")

if __name__ == "__main__":
    main()