# load results and plot
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os
import spacepy.pycdf as pycdf
import sys
sys.path.append('../ml_decay_prediction/react_thesis')
from traj_predict_tle import read_sw_nrlmsise00, get_sw_params 
import datetime as dt
import pandas as pd
import matplotlib.gridspec as gridspec
from combine_lat_lst_plot import reconstruct_lat_lst_time
from scipy.interpolate import griddata
from compare_recon_and_swarm import load_swarm_dens
import numpy as np
from scipy.ndimage import distance_transform_edt
def main():
    # open pkl file with results
    # === Load results from pkl file
    t_min = 5
    file_path = 'data/gannon_20240509_' + str(t_min) + 'm_ds.pkl'
    # file_path = 'data/octG4_20241006_' + str(t_min) + 'm_ds.pkl'
    # file_path = 'data/gannon_20240509_' + str(t_min) + 'm_ds.pkl'
    ref_dir = file_path[:-4] + "_2d_owd"
    output_dir = file_path[:-4] + "_lat_lst_combined"
    # make output dir if it doesnt exist
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(ref_dir):
        print(f"Output directory {output_dir} does not exist.")

    with open(f"{ref_dir}/results.pkl", 'rb') as f:
        results = pkl.load(f)
    tvec = results['tvec']
    rho0_vec_lat = results['rho0_vec']
    H_vec_lat = results['H_vec']
    lat_counts = results['counts']
    lat_bins = np.linspace(-90, 90, rho0_vec_lat.shape[1] + 1)  # assuming lat_bins are in results


    h0 = 600

    ref_dir = file_path[:-4] + "_2D_lst_owd"
    if not os.path.exists(ref_dir):
        print(f"Output directory {ref_dir} does not exist.")

    with open(f"{ref_dir}/results.pkl", 'rb') as f:
        results = pkl.load(f)
    tvec = results['tvec']
    rho0_vec_lst = results['rho0_vec']
    H_vec_lst = results['H_vec']
    lst_counts = results['counts']
    lst_bins = np.linspace(0, 24, rho0_vec_lst.shape[1] + 1)  # assuming lat_bins are in results

    # compute dens at 500 km altitude for lat and lst
    # beta_avg = 8e-8 #6e-8 # 1.8e-7
    # alt_query = 470
    # dens_2d_lat = (1/beta_avg) * rho0_vec_lat.T * np.exp(-(alt_query-h0) / H_vec_lat.T) / 1e9 # convert rho0 to kg/m³
    # dens_2d_lst = (1/beta_avg) * rho0_vec_lst.T * np.exp(-(alt_query-h0) / H_vec_lst.T) / 1e9 # convert rho0 to kg/m³

    H_vec = reconstruct_lat_lst_time(H_vec_lat.T, H_vec_lst.T)
    rho0_vec = reconstruct_lat_lst_time(rho0_vec_lat.T, rho0_vec_lst.T)
    counts = reconstruct_lat_lst_time(lat_counts.T, lst_counts.T)

    # remove values with counts < 20
    # valid_mask = counts >= 60
    # # set invalid values to nan
    # rho0_vec[~valid_mask] = np.nan
    # H_vec[~valid_mask] = np.nan

    # # interpolate missing values using gaussian
    # x, y, z = np.meshgrid(np.arange(rho0_vec.shape[0]), np.arange(rho0_vec.shape[1]), np.arange(rho0_vec.shape[2]), indexing='ij')
    # rho0_vec = griddata((x[valid_mask], y[valid_mask], z[valid_mask]), rho0_vec[valid_mask], (x, y, z), method='nearest')
    # H_vec = griddata((x[valid_mask], y[valid_mask], z[valid_mask]), H_vec[valid_mask], (x, y, z), method='nearest')

    # counts = counts[valid_mask]

    # X = (1/beta_avg) * rho0_vec * np.exp(-(alt_query-h0) / H_vec) / 1e9 # convert rho0 to kg/m³

    #     # open the pickle file
    # output_dir = "output_lat_lst_time_fit"
    # file_path = "dens_results_lat_lst_time_june2015_30m_weighted.pkl"
    # # file_path = "dens_results_lat_lst_time.pkl"
    # with open(f"{output_dir}/{file_path}", 'rb') as f:
    #     results = pkl.load(f)

    # tvec = results['tvec']
    # lat_bins = results['lat_bins']
    # lst_bins = results['lst_bins']
    # rho0_vec = results['rho0_vec']
    # H_vec = results['H_vec']
    # loss_history = results['loss_history']
    # beta_vec = results['beta_vec']
    # counts = results['counts']

    # for i in range(len(tvec)):
    #     plt.figure()
    #     plt.imshow(counts[i,:,:], label='Counts', aspect='auto', extent=[lst_bins[0], lst_bins[-1], lat_bins[0], lat_bins[-1]], origin='lower', cmap='jet')
    #     plt.colorbar(label='Counts')
    #     plt.title(f'Time {tvec[i]}s')
    #     plt.tight_layout()
    #     plt.show()

    # # Step 1: Identify valid and invalid indices
    valid_mask = counts >= 70
    invalid_mask = ~valid_mask

    # convert all invalid values to nan
    rho0_vec[invalid_mask] = np.nan
    H_vec[invalid_mask] = np.nan

    # rho0_vec = fill_invalid_with_nearest(rho0_vec, valid_mask=valid_mask)
    # H_vec = fill_invalid_with_nearest(H_vec, valid_mask=valid_mask)

    # # for indices where invalid_mask is True, find the nearest valid cell and copy its value in the invalid cell
    # for i in range(len(tvec)):
    #     for j in range(len(lat_bins)-1):
    #         for k in range(len(lst_bins)-1):
    #             if invalid_mask[i,j,k]:
    #                 # Find the nearest valid cell
    #                 nearest_valid = np.argmin(np.abs(np.nonzero(valid_mask[i,j,k])[0] - k))
    #                 # Copy the value from the nearest valid cell
    #                 rho0_vec[i,j,k] = rho0_vec[i,j,nearest_valid]
    #                 H_vec[i,j,k] = H_vec[i,j,nearest_valid]

    # valid_indices = np.array(np.nonzero(valid_mask)).T
    # invalid_indices = np.array(np.nonzero(invalid_mask)).T

    # # Step 2: Build KDTree from valid cells
    # tree = cKDTree(valid_indices)

    # # Step 3: Find nearest valid cell for each invalid cell
    # _, nearest_idx = tree.query(invalid_indices)

    # # Step 4: Replace rho and H in invalid cells with nearest valid values
    # rho0_vec[invalid_mask] = rho0_vec[tuple(valid_indices[nearest_idx].T)]
    # H_vec[invalid_mask] = H_vec[tuple(valid_indices[nearest_idx].T)]

#     # Create coordinate grid
#     x, y, z = np.meshgrid(np.arange(len(lat_bins)-1), np.arange(len(lst_bins)-1), np.arange(len(tvec)), indexing='ij')

#    # Flatten the coordinate arrays and mask
#     x_flat = x.flatten()
#     y_flat = y.flatten()
#     z_flat = z.flatten()
#     valid_mask_flat = valid_mask.flatten()

#     # Only use valid points for interpolation
#     points_valid = np.column_stack((x_flat[valid_mask_flat], y_flat[valid_mask_flat], z_flat[valid_mask_flat]))
#     rho_values_valid = rho0_vec.flatten()[valid_mask_flat]
#     H_values_valid = H_vec.flatten()[valid_mask_flat]

#     # Interpolate onto the full grid
#     points_full = np.column_stack((x_flat, y_flat, z_flat))
#     rho_interp = griddata(points_valid, rho_values_valid, points_full, method='nearest').reshape(x.shape)
#     H_interp = griddata(points_valid, H_values_valid, points_full, method='nearest').reshape(x.shape)

#     # Fill invalid cells
#     rho0_vec = rho_interp
#     H_vec = H_interp

    # load swarm density data
    dir = '../ml_decay_prediction/react_thesis/swarm_data_C_pod_gannon'
    swarm_dens, swarm_time, swarm_lat, swarm_lon, swarm_lst, swarm_alt = load_swarm_dens(dir)

    # compute density based on rho0_vec and H_vec for each time, lat, lst, and alt from swarm
    rho_recon = np.zeros(len(swarm_time))
    time_idx = np.zeros(len(swarm_time), dtype=int)
    lat_idx = np.zeros(len(swarm_time), dtype=int)
    lst_idx = np.zeros(len(swarm_time), dtype=int)

    lat_idx = np.digitize(swarm_lat, lat_bins) - 1
    lst_idx = np.digitize(swarm_lst, lst_bins) - 1
    rho0 = np.zeros(len(swarm_time))
    H = np.zeros(len(swarm_time))

    for i in range(len(swarm_time)):
        # find the closest time in tvec
        time_idx[i] = np.argmin(np.abs(tvec - swarm_time[i]))

        rho0[i] = rho0_vec[lat_idx[i], lst_idx[i], time_idx[i]]
        H[i] = H_vec[lat_idx[i], lst_idx[i], time_idx[i]]
        # compute the density
        rho_recon[i] = rho0[i] * np.exp(-(swarm_alt[i] - 600) / H[i])

    # compute the scale factor to match the swarm density
    i_s = 2000
    i_e = -5000
    a = np.nansum(rho_recon[i_s:i_e] * swarm_dens[i_s:i_e]) / np.nansum(rho_recon[i_s:i_e] ** 2)


    plt.figure()
    plt.plot(swarm_time, swarm_dens, alpha = 0.25, label='Swarm POD Density', color = 'black')
    plt.plot(swarm_time[i_s:i_e], a*rho_recon[i_s:i_e], alpha = 0.5, label='REACT', color = 'tab:blue')
    # print a in the corner
    # plt.text(0.05, 0.95, f'Scaling Factor: {a:.2f}', transform=plt.gca().transAxes, fontsize=12)
    plt.ylabel('Density (kg/m³)')
    plt.legend()
    plt.xlabel('Date (UTC)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    # Plot density
    ax1.plot(swarm_time, swarm_dens, alpha=0.5, label='Swarm Density')
    ax1.plot(swarm_time, rho_recon, alpha=0.5, label='Reconstructed Density')
    ax1.set_ylabel('Density (kg/m³)')
    ax1.legend()
    # Plot latitude
    ax2.plot(swarm_time, swarm_lat, alpha=0.5)
    ax2.set_ylabel('Latitude (°)')
    # Plot local solar time
    ax3.plot(swarm_time, swarm_lst, alpha=0.5)
    ax3.set_ylabel('Local Solar Time (°)')
    ax3.set_xlabel('Time')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.show()

    # plot the lat vs lon history (one plot per timestep)
    ref_alt = 600
    dens_full_recon = np.zeros((len(tvec), len(lat_bins)-1, len(lst_bins)-1))
    for i in range(len(tvec)):
        dens_full_recon[i, :, :] = rho0_vec[i, :, :] * a * np.exp(-(ref_alt - 600) / H_vec[i, :, :])

    # convert LST to longitude
    lon_bins = np.linspace(-180, 180, len(lst_bins))  # you can define more if needed

    #define utc_hour_list -- the list of UTC hours for each time step
    utc_hour_list = np.array([t.hour + t.minute / 60.0 + t.second / 3600.0 for t in tvec])

    # Output:
    dens_full_recon_lon = np.zeros((len(utc_hour_list), len(lat_bins)-1, len(lon_bins)-1))

    # === Main loop ===
    for t_idx, utc_hour in enumerate(utc_hour_list):
        # shift
        slice = dens_full_recon[t_idx, :, :]
        # compute difference 
        diff_deg = utc_hour * (360 / 24)
        deg_per_bin = lon_bins[1] - lon_bins[0]
        # shift the entries in slice back by np.floor(diff_deg/deg_per_bin), but if the shift becomes negative, wrap it around to the other side
        shift_bins = int(np.floor(diff_deg / deg_per_bin))
        c = 3
        # take the slice and shift it in the second dimension (longitude) by shift_bins
        dens_full_recon_lon[t_idx, :, :] = np.roll(slice, -shift_bins, axis=1)
        # if shift_bins < 0:
        #     shift_bins += len(lon_bins)


    maxdens = np.nanmax(dens_full_recon)
    mindens = np.nanmin(dens_full_recon)
    # convert lst to longitude. Each timestep, reference it to the local solar time
    idx_sample = np.linspace(0, len(tvec)-1, 100, dtype=int)
    # plt.figure(figsize=(12, 6))
    for i in idx_sample:
        plt.figure(figsize=(12, 6))
        plt.imshow(dens_full_recon_lon[i, :, :], aspect='auto', extent=[lon_bins[0], lon_bins[-1], lat_bins[0], lat_bins[-1]], origin='lower', cmap='jet')
        plt.colorbar(label='Density (kg/m³)')
        plt.xticks(rotation=45)
        plt.xlabel('Longitude (°)')
        plt.ylabel('Latitude (°)')
        plt.title(f'Time {tvec[i]}s')
        plt.clim(mindens, maxdens)  # set color limits
        # save figs
        plt.tight_layout()
        plt.savefig(f"figs_0730_june2015/dens_lat_lon_{i:04d}.png")
        plt.close()
        print(i/len(tvec))



    rho0_avg = np.nanmean(results['rho0_vec'], axis=(1, 2))
    H_avg = np.nanmean(results['H_vec'], axis=(1, 2))

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(results['tvec'], rho0_avg)
    plt.title('Average rho0(t)')
    plt.xlabel('Time')
    plt.ylabel('Density (kg/m³)')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(results['tvec'], H_avg)
    plt.title('Average H(t)')
    plt.xlabel('Time')
    plt.ylabel('Scale Height (m)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # plot the results
    # maxcounts = np.max(counts)
    # for i in range(len(tvec)):
    #     plt.figure()
    #     plt.imshow(counts[:,:,i], aspect='auto', extent=[lat_bins[0], lat_bins[-1], lst_bins[0], lst_bins[-1]], origin='lower')
    #     plt.colorbar(label='Counts')
    #     plt.clim(0, maxcounts)  # set color limits
    #     plt.xlabel('Latitude (°)')
    #     plt.ylabel('LST (°)')
    #     plt.title(f'Counts of samples {tvec[i]}s')
    #     plt.show()


    # plot the results
    maxrho0 = np.max(rho0_vec)
    maxH = np.max(H_vec)
    minH = np.min(H_vec)
    lent = len(tvec)
    # sample 10 indices
    sample_indices = np.linspace(0, lent-1, 10, dtype=int)
    j = 1
    plt.figure()
    for i in sample_indices:
        plt.subplot(5,2,j)
        plt.imshow(rho0_vec[i,:,:].T, aspect='auto', extent=[lat_bins[0], lat_bins[-1], lst_bins[0], lst_bins[-1]], origin='lower')
        plt.colorbar(label='rho0 (kg/m³)')
        # plt.clim(0, maxrho0)  # set color limits
        plt.xlabel('Latitude (°)')
        plt.ylabel('LST')
        # plt.title(f'Density (rho0) at Time {tvec[i]}s')
        j += 1

    j = 1
    plt.figure()
    for i in sample_indices:
        plt.subplot(5,2,j)
        plt.imshow(H_vec[i,:,:].T, aspect='auto', extent=[lat_bins[0], lat_bins[-1], lst_bins[0], lst_bins[-1]], origin='lower')
        plt.colorbar(label='H (m)')
        # plt.clim(minH, maxH)  # set color limits
        plt.xlabel('Latitude (°)')
        plt.ylabel('LST')
        # plt.title(f'Scale Height (H) at Time {tvec[i]}s')
        # plt.tight_layout()                                        
        j += 1

    plt.show()



def fill_invalid_with_nearest(arr, *, valid_mask=None, invalid_mask=None, sampling=None, copy=True):
    """
    Fill invalid cells of a 3D array with the value from the nearest valid cell.

    Parameters
    ----------
    arr : np.ndarray
        3D array of values to fill.
    valid_mask : np.ndarray, optional
        Boolean mask (True where values are valid). Shape must match `arr`.
        Provide exactly one of `valid_mask` or `invalid_mask`.
    invalid_mask : np.ndarray, optional
        Boolean mask (True where values are invalid). Shape must match `arr`.
        Provide exactly one of `valid_mask` or `invalid_mask`.
    sampling : tuple(float, float, float), optional
        Per-axis spacing used for the distance metric, e.g. (dt, dlat, dlst).
        Defaults to 1.0 for each axis if not given.
    copy : bool, default True
        If True, operate on a copy and return it. If False, fill `arr` in place.

    Returns
    -------
    filled : np.ndarray
        Array with invalid cells replaced by nearest valid values.

    Notes
    -----
    - Uses Euclidean distance with optional anisotropic scaling via `sampling`.
    - Works for any ndim, but intended for 3D here.
    """
    if (valid_mask is None) == (invalid_mask is None):
        raise ValueError("Provide exactly one of valid_mask or invalid_mask.")

    if invalid_mask is None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        invalid_mask = ~valid_mask
    else:
        invalid_mask = np.asarray(invalid_mask, dtype=bool)
        valid_mask = ~invalid_mask

    if arr.shape != valid_mask.shape:
        raise ValueError("arr and mask shapes must match.")

    if not np.any(valid_mask):
        raise ValueError("No valid cells to copy from.")

    # distance_transform_edt computes distances to zeros; we want zeros at valid cells.
    # So pass `~valid_mask` as the 'ones', making valid cells the background (zeros).
    # return_indices gives, for every voxel, the indices of the closest zero (valid) voxel.
    _, nearest_idx = distance_transform_edt(~valid_mask, return_indices=True, sampling=sampling)

    # Prepare output
    out = arr.copy() if copy else arr

    # For invalid locations, pull the value from the nearest valid coordinates
    inv = invalid_mask
    coords = tuple(nearest_idx[d][inv] for d in range(arr.ndim))
    out[inv] = arr[coords]

    return out


if __name__ == "__main__":
    main()