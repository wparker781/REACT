# read dens data
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime

# Path to the uploaded file
directory = "wam_ipe_dens_gannon"

# find all the .nc files in the directory
files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.nc')]

file_dt = []
max_dens_file = []
for file in files:
    # Extract the datetime part (8 digits + underscore + 6 digits)
    match = re.search(r'(\d{8}_\d{6})', file)
    if match:
        dt_str = match.group(1)
        # Parse into datetime object
        dt = datetime.strptime(dt_str, "%Y%m%d_%H%M%S")
        parsed_datetime = dt
    else:
        parsed_datetime = None

    # get max dens
    ds = xr.open_dataset(file)
    den_400 = ds['den'].sel(hlevs=400.0, method='nearest').squeeze()
    max_dens_file.append(den_400.max().item())
    file_dt.append(parsed_datetime)

# Sort files by datetime
files = [f for _, f in sorted(zip(file_dt, files))]
file_dt = sorted(file_dt)
max_dens = max(max_dens_file)
# Open the NetCDF file using xarray

# get indices for every 10th file to plot
# file_plot_idx = np.arange(0, len(files), 10)
file_plot_idx = np.linspace(0, len(files)-1, 100).astype(int)
for i in range(len(file_plot_idx)):
    ds = xr.open_dataset(files[file_plot_idx[i]])

    # Select density at 400 km altitude
    den_400 = ds['den'].sel(hlevs=400.0, method='nearest').squeeze()

    # Shift longitude from 0–360 to -180–180
    den_400_shifted = den_400.roll(lon=den_400.shape[0]//2, roll_coords=True)
    lon_shifted = ((ds['lon'] + 180) % 360) - 180

    # Plot shifted data
    plt.figure(figsize=(10,5))
    im = plt.imshow(den_400_shifted, origin='lower',
                    extent=[lon_shifted.min(), lon_shifted.max(), ds['lat'].min(), ds['lat'].max()],
                    aspect='auto', cmap='jet')
    plt.colorbar(im, label='Density')
    plt.clim(0, max_dens)
    plt.title(f'Density at 400 km from {file_dt[file_plot_idx[i]]}')
    plt.xlabel('Longitude (°)')
    plt.ylabel('Latitude (°)')
    plt.show()