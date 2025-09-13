import numpy as np
from scipy.sparse import diags, vstack, csr_matrix
from scipy.sparse.linalg import lsqr
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d, uniform_filter
import pickle as pkl
import os
from tqdm import tqdm
import time
import pandas as pd
# set font to arial
plt.rcParams['font.family'] = 'Arial'

# === Main Function ===
def main():
    t0 = time.time()
    # === Load Data ===
    t_min = 10
    # file_path = 'density_history_gannon_v2_' + str(t_min) + 'm_filt.pkl'
    file_path = 'data/gannon_20240509_' + str(t_min) + 'm_ds.pkl'
    with open(file_path, 'rb') as f:
        data = pkl.load(f)

    output_dir = "output_lat_lst_time_fit"
    os.makedirs(output_dir, exist_ok=True)

    weighting = True

    ds_list = np.array(data['ds'])

    # valid = np.where((ds_list > 1e-15) & (ds_list < 1e-10))  # filter out large/small ds for stability
    # ds_list = ds_list[valid]
    h = np.array(data['alt'])
    lon_list = data['lon']
    len_list = np.array([len(d) for d in lon_list])
    valid = np.where(len_list > 2)  # filter out empty measurements
    ds_list = ds_list[valid]
    h = h[valid]
    time_list = [data['time'][i] for i in valid[0]]  # list of lists
    lat_list = [data['lat'][i] for i in valid[0]]
    lon_list = [data['lon'][i] for i in valid[0]]
    # plt.figure()
    # plt.hist(len_list, bins=100)
    # plt.title('Distribution of number of interpolated steps per measurement')
    # plt.show()
    satcat_list = [data['satcat'][i] for i in valid[0]]
    tvec = data['tvec']
    # make a new tvec that is every 5th element
    # tvec = tvec[::5]
    m = len(ds_list)
    nt = len(tvec)

    print(np.min(ds_list), np.max(ds_list))

    # plt.figure()
    # plt.semilogy(ds_list)
    # plt.show()


    # === Latitude binning ===
    lat_bins = np.linspace(-90, 90, 20)
    lon_bins = np.linspace(-180, 180, 20)  # 10 degree bins
    nlat = len(lat_bins) - 1
        # convert to LST
    lst_list = []
    for i in range(m):
        dt_index = pd.DatetimeIndex(time_list[i])
        utc_hours = dt_index.hour + dt_index.minute / 60.0 + dt_index.second / 3600.0
        lst_list.append((lon_list[i] / 15.0 + utc_hours) % 24)

    lst_bins = np.linspace(0, 24, 20)
    nlst = len(lst_bins) - 1

    # === Flatten and build full measurement array ===
    # d_s_all, h_all, time_idx_all, lat_idx_all, sat_idx_all = [], [], [], [], []

    unique_satcats, satcat_indices = np.unique(satcat_list, return_inverse=True)
    n_satcat = len(unique_satcats)

    # compute time indices and lat indices for each measurment in time_list and lat_list
    time_idx = []
    lat_idx = []
    lst_idx = []
    lon_idx = []
    for i in range(m):
        time_idx.append(np.searchsorted(tvec, time_list[i]))
        lat_idx.append(np.digitize(lat_list[i], lat_bins) - 1)
        lst_idx.append(np.digitize(lst_list[i], lst_bins) - 1)
        lon_idx.append(np.digitize(lon_list[i], lon_bins)-1)


    # compute counts in each lst x time bin
    counts_lst = np.zeros((nt, nlst))
    for i in range(m):
        time_bin = time_idx[i]
        lst_bin = lst_idx[i]
        counts_lst[time_bin, lst_bin] += 1

    # repeat for longitude
    counts_lon = np.zeros((nt, nlst))
    for i in range(m):
        time_bin = time_idx[i]
        lon_bin = lon_idx[i]
        counts_lon[time_bin, lon_bin] += 1

    # plot 
    plt.figure(figsize=(8, 4))
    ax1 = plt.subplot(2, 1, 1)
    plt.imshow(counts_lon.T, aspect='auto', origin='lower', cmap='viridis', extent = [tvec[0], tvec[-1], -180, 180], interpolation= 'none')
    plt.colorbar(label='Counts')
    plt.title('Longitude bins')
    # plt.xlabel('Date (UTC)')
    plt.ylabel('Longitude (degrees)')
    ax1.set_xticklabels([])
    ax2 = plt.subplot(2,1, 2)
    plt.imshow(counts_lst.T, aspect='auto', origin='lower', cmap='viridis', extent = [tvec[0], tvec[-1], 0, 24], interpolation='none')
    plt.colorbar(label='Counts')
    plt.title('LST bins')
    plt.xlabel('Date (UTC)')
    plt.ylabel('LST (hours)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()




    




if __name__ == "__main__":
    main()