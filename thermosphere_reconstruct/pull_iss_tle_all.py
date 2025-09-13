# pip install spacetrack
from spacetrack import SpaceTrackClient
from spacetrack.operators import between
import time
from datetime import datetime

# ask for username and password
USER = input("Enter your Space-Track username: ")
PASS = input("Enter your Space-Track password: ")

norad = 25544  # ISS (ZARYA)
first_year = 1998
last_year = datetime.utcnow().year + 1  # include current year

st = SpaceTrackClient(identity=USER, password=PASS)

all_parts = []
for yr in range(first_year, last_year):
    start = f"{yr}-01-01"
    end   = f"{yr+1}-01-01"
    print(f"Fetching {yr}…")
    tle_text = st.gp_history(
        norad_cat_id=norad,
        epoch=between(start, end),
        orderby="EPOCH asc",
        format="tle",             # TLE text (you can also use format="csv" or "json")
    )
    if tle_text.strip():
        with open(f"iss_25544_{yr}.tle", "w") as f:
            f.write(tle_text if tle_text.endswith("\n") else tle_text + "\n")
        all_parts.append(tle_text if tle_text.endswith("\n") else tle_text + "\n")
    time.sleep(1)  # be polite re: rate limits

# Merge to a single file
with open("iss_25544_all.tle", "w") as f:
    for chunk in all_parts:
        f.write(chunk)

print("Done. Files written: per-year TLEs and iss_25544_all.tle")
