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
from pymsis import msis
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime

def main():
    # open pkl file with results
    # === Load results from pkl file
    t_min = 5
    file_path = 'data/gannon_20240509_' + str(t_min) + 'm_ds.pkl'
    # file_path = 'data/octG4_20241006_' + str(t_min) + 'm_ds.pkl'
    # file_path = 'data/gannon_20240509_' + str(t_min) + 'm_ds.pkl'
    ref_dir = file_path[:-4] + "_2d_owd"
    output_dir = file_path[:-4] + "_lat_lst_combined_compare"
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
    lst_bins = np.linspace(0, 24, rho0_vec_lat.shape[1] + 1)  # assuming lat_bins are in results

    # compute dens at 400 km altitude for lat and lst
    beta_avg = 2.5e-8 # 1.8e-7
    alt_query = 400
    # dens_2d_lat = (1/beta_avg) * rho0_vec_lat.T * np.exp(-(alt_query-h0) / H_vec_lat.T) / 1e9 # convert rho0 to kg/m³
    # dens_2d_lst = (1/beta_avg) * rho0_vec_lst.T * np.exp(-(alt_query-h0) / H_vec_lst.T) / 1e9 # convert rho0 to kg/m³

    X_H = reconstruct_lat_lst_time(H_vec_lat.T, H_vec_lst.T)
    X_rho0 = reconstruct_lat_lst_time(rho0_vec_lat.T, rho0_vec_lst.T)

    X = (1/beta_avg) * X_rho0 * np.exp(-(alt_query-h0) / X_H) / 1e9 # convert rho0 to kg/m³

    # load ap
    # also open space weather data csv
    # set tvec to be every hour from start date to end date
    start_date = min(tvec)#dt.datetime(2024, 5, 9, 0, 0)
    end_date = max(tvec)#dt.datetime(2024, 5, 15, 0, 0)
    # tvec = np.arange(start_date, end_date, dt.timedelta(hours=1))
    t_dt = pd.to_datetime(tvec)
    # Check to see if appropriate sw file already exists
    if os.path.exists('data/sw_data_'+start_date.strftime('%d_%m_%Y')+'_'+ end_date.strftime('%d_%m_%Y')+'.pkl'):
        with open('data/sw_data_'+start_date.strftime('%d_%m_%Y')+'_'+ end_date.strftime('%d_%m_%Y')+'.pkl', 'rb') as f:
            f107A, f107, Ap, aph, t_sw = pkl.load(f)
    else:
        print('loading sw data...')
        sw_data = read_sw_nrlmsise00('data/SW-All.csv')
        f107A, f107, Ap, aph = get_sw_params(tvec, sw_data, 0, 0)
        with open('data/sw_data_'+start_date.strftime('%d_%m_%Y')+ '_'+ end_date.strftime('%d_%m_%Y')+'.pkl', 'wb') as f:
            pkl.dump([f107A, f107, Ap, aph, tvec], f)
    print('loaded all data!')

    # load WAM-IPE results
    # read dens data
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
    # max_dens = max(max_dens_file)
    # Open the NetCDF file using xarray

    # get indices for every 10th file to plot
    # file_plot_idx = np.arange(0, len(files), 10)
    file_plot_idx = np.linspace(0, len(files)-1, 100).astype(int)
    den_wam_ipe = np.zeros((len(file_plot_idx), 91,90))
    for i in range(len(file_plot_idx)):
        ds = xr.open_dataset(files[file_plot_idx[i]])

        # Select density at 400 km altitude
        den_400 = ds['den'].sel(hlevs=400.0, method='nearest').squeeze()

        # Shift longitude from 0–360 to -180–180
        den_400_shifted = den_400.roll(lon=den_400.shape[0]//2, roll_coords=True)
        den_wam_ipe[i,:,:] = den_400_shifted.values
        lon_shifted = ((ds['lon'] + 180) % 360) - 180

        # Plot shifted data
        # plt.figure(figsize=(10,5))
        # im = plt.imshow(den_400_shifted, origin='lower',
        #                 extent=[lon_shifted.min(), lon_shifted.max(), ds['lat'].min(), ds['lat'].max()],
        #                 aspect='auto', cmap='jet')
        # plt.colorbar(im, label='Density')
        # plt.clim(0, max_dens)
        # plt.title(f'Density at 400 km from {file_dt[file_plot_idx[i]]}')
        # plt.xlabel('Longitude (°)')
        # plt.ylabel('Latitude (°)')
        # plt.show()

    # shift X to switch LST to longitude
    # lst_bins are in hours, convert to degrees
    # lst_bins_deg = lst_bins * 15
    # lon_bins = (lst_bins_deg - 180) % 360 - 180
    # lon_bins = np.roll(lon_bins, len(lon_bins)//2)
    # roll X according to tvec relative to 00 UTC
    for i in range(len(t_dt)):
        X[:, :, i] = np.roll(X[:, :, i], -int(t_dt[i].hour * (X.shape[1] / 24))+len(lst_bins)//2, axis=1)
    lon_bins = np.linspace(-180, 180, X.shape[1] + 1)


    # plot msis2.0 density at 500 km for lat and lst
    # dates = np.arange(np.datetime64("2024-05-09T00:00"), np.datetime64("2024-05-15T00:00"), np.timedelta64(100, "m"))
    lat = np.linspace(-90,90,30)
    lon = np.linspace(-180,180,30)
    # geomagnetic_activity=-1 is a storm-time run
    data = msis.run(tvec, lon, lat, alt_query, geomagnetic_activity=-1)

    #

    # plot the reconstructed density
    max_dens = np.nanmax(X)
    idx_05_10 = np.where((t_dt.month == 5) & (t_dt.day == 10))[0][0]
    idx_05_14 = np.where((t_dt.month == 5) & (t_dt.day == 14))[0][0]
    idx_plot = np.linspace(idx_05_10, idx_05_14, 100).astype(int)
    # idx_plot = np.linspace(0, X.shape[2] - 1, 100).astype(int)

    max_msis_dens = np.nanmax(data[:, :, :, 0, 0])
    max_wam_dens = np.nanmax(den_wam_ipe)
    # for i in idx_plot:
    #     plt.figure(figsize=(6.5, 5))
    #     plt.imshow(data[i, :, :, 0, 0].T, extent=[-180, 180, -90, 90], origin='lower', aspect='auto', cmap = 'jet', interpolation='gaussian')
    #     plt.title(f'MSIS2.0 Density at {t_dt[i].month}/{t_dt[i].day} {t_dt[i].hour}:{t_dt[i].minute:02d}')
    #     plt.colorbar(label='Density (kg/m³)')
    #     plt.clim(0, max_msis_dens)  # Adjust as needed for better visibility
    #     plt.xlabel('Longitude [°]')
    #     plt.ylabel('Latitude [°]')
    #     plt.grid(axis = 'both', linestyle = '-', color = 'white', alpha = 0.3, linewidth =0.5)
    #     plt.show()


    j = 0
    for i in idx_plot:
        fig = plt.figure(figsize=(12, 7))
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1])

        # Top left: WAM-IPE density
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(den_wam_ipe[j,:,:], extent=[-180, 180, -90, 90], origin='lower',
                        aspect='auto', cmap='jet', interpolation='gaussian')
        fig.colorbar(im1, ax=ax1, label='Density (kg/m³)')
        ax1.set_title(f'WAM-IPE: {t_dt[i].month}/{t_dt[i].day} {t_dt[i].hour}:{t_dt[i].minute:02d}')
        ax1.set_ylabel('Latitude [°]')
        ax1.set_xlabel('Longitude [°]')
        ax1.grid(axis='both', linestyle='-', color='white', alpha=0.3, linewidth=0.5)
        im1.set_clim(0, max_wam_dens)

        # Top right: Reconstructed density
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(X[:, :, i], aspect='auto', origin='lower',
                        extent=[lon_bins[0], lon_bins[-1], lat_bins[0], lat_bins[-1]],
                        cmap='jet', interpolation='gaussian')
        fig.colorbar(im2, ax=ax2, label='Density (kg/m³)')
        ax2.set_title(f'REACT: {t_dt[i].month}/{t_dt[i].day} {t_dt[i].hour}:{t_dt[i].minute:02d}')
        ax2.set_ylabel('Latitude [°]')
        ax2.set_xlabel('Longitude [°]')
        ax2.grid(axis='both', linestyle='-', color='white', alpha=0.3, linewidth=0.5)
        im2.set_clim(0, max_dens)

        # Bottom: Space weather parameters (shared for both)
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(t_dt, aph, color='darkcyan')
        ax3.set_ylabel('ap', color='darkcyan')
        ax3.tick_params(axis='y', labelcolor='darkcyan')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(t_dt, f107, color='darkorange', alpha=0.5)
        ax3_twin.set_ylabel(r'F$_{10.7}$ [sfu]', color='darkorange')
        ax3_twin.tick_params(axis='y', labelcolor='darkorange')
        ax3.set_xlabel('Date (UTC)')
        ax3.set_title('Space Weather Parameters')
        ax3.axvline(t_dt[i], color='r', linestyle='--', linewidth=2)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/reconstructed_density_0902_compare_wam_{i:03d}.png", dpi=100)
        plt.close()
        print(i/X.shape[2])

        j += 1

        #------

    for i in idx_plot:
        fig = plt.figure(figsize=(6.5, 5))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        
        ax1 = fig.add_subplot(gs[0])
        im = ax1.imshow(data[i, :, :, 0, 0].T, extent=[-180, 180, -90, 90], origin='lower', aspect='auto', cmap = 'jet', interpolation='gaussian')
        fig.colorbar(im, ax=ax1, label='Density (kg/m³)')
        ax1.set_title(f'MSIS2: {t_dt[i].month}/{t_dt[i].day} {t_dt[i].hour}:{t_dt[i].minute:02d}')
        ax1.set_ylabel('Latitude [°]')
        ax1.set_xlabel('Longitude [°]')
        ax1.grid(axis='both', linestyle='-', color='white', alpha=0.3, linewidth=0.5)
        im.set_clim(0, max_msis_dens)
        
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(t_dt, aph, color='darkcyan')
        ax2.set_ylabel('ap', color='darkcyan')
        ax2.tick_params(axis='y', labelcolor='darkcyan')
        plt.xticks(rotation=45)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(t_dt, f107, color='darkorange', alpha=0.5)
        ax2_twin.set_ylabel(r'F$_{10.7}$ [sfu]', color='darkorange')
        ax2_twin.tick_params(axis='y', labelcolor='darkorange')
        ax2.set_xlabel('Date (UTC)')
        # ax2.set_title('Space Weather Parameters')
        ax2.axvline(t_dt[i], color='r', linestyle='--', linewidth=2)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/reconstructed_density_0902_msis_{i:03d}.png", dpi=300)
        plt.close()
        print(i/X.shape[2])

    
    for i in idx_plot:
        fig = plt.figure(figsize=(6.5, 5))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        
        ax1 = fig.add_subplot(gs[0])
        im = ax1.imshow(X[:, :, i], aspect='auto', origin='lower',
                        extent=[lon_bins[0], lon_bins[-1], lat_bins[0], lat_bins[-1]],
                        cmap='jet', interpolation='gaussian')
        fig.colorbar(im, ax=ax1, label='Density (kg/m³)')
        ax1.set_title(f'{t_dt[i].month}/{t_dt[i].day} {t_dt[i].hour}:{t_dt[i].minute:02d}')
        ax1.set_ylabel('Latitude [°]')
        ax1.set_xlabel('Longitude [°]')
        ax1.grid(axis='both', linestyle='-', color='white', alpha=0.3, linewidth=0.5)
        im.set_clim(0, max_dens)
        
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(t_dt, aph, color='darkcyan')
        ax2.set_ylabel('ap', color='darkcyan')
        ax2.tick_params(axis='y', labelcolor='darkcyan')
        plt.xticks(rotation=45)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(t_dt, f107, color='darkorange', alpha=0.5)
        ax2_twin.set_ylabel(r'F$_{10.7}$ [sfu]', color='darkorange')
        ax2_twin.tick_params(axis='y', labelcolor='darkorange')
        ax2.set_xlabel('Date (UTC)')
        # ax2.set_title('Space Weather Parameters')
        ax2.axvline(t_dt[i], color='r', linestyle='--', linewidth=2)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/reconstructed_density_0902_{i:03d}.png", dpi=300)
        plt.close()
        print(i/X.shape[2])


    # # plot the reconstructed density
    # max_dens = np.nanmax(X)
    # idx_plot = np.linspace(0, X.shape[2] - 1, 100).astype(int)
    # for i in idx_plot:
    #     plt.figure(figsize=(5, 3))
    #     plt.imshow(X[:, :, i], aspect='auto', origin='lower', extent=[lst_bins[0], lst_bins[-1], lat_bins[0], lat_bins[-1]], cmap='jet', interpolation = 'gaussian')
    #     plt.colorbar(label='Density (kg/m³)')
    #     plt.title(f'{t_dt[i].month}/{t_dt[i].day} {t_dt[i].hour}:{t_dt[i].minute:02d}')
    #     plt.ylabel('Latitude [°]')
    #     plt.xlabel('LST [hours]')
    #     plt.clim(0, max_dens)
    #     plt.tight_layout()
    #     plt.savefig(f"{output_dir}/reconstructed_density_{i:03d}.png", dpi=300)
    #     plt.show()
    #     print(i/X.shape[2])
    # c = 3


 


def reconstruct_lat_lst_time(
    dens_lat,              # shape: (nlat, nt)  average over LST for each lat, per time
    dens_lst,              # shape: (nlst, nt)  average over lat for each LST, per time
    w_lat=None,            # optional lat-bin widths (nlat,), default=1
    w_lst=None,            # optional LST-bin widths (nlst,), default=1
    prior=None,            # optional prior P of shape (nlat, nlst, nt) or None
    max_iter=2000,
    tol=1e-7,
    eps=1e-12
):
    """
    Returns X of shape (nlat, nlst, nt) such that:
      (1) row-averages over LST match dens_lat[:, t]
      (2) column-averages over lat match dens_lst[:, t]
    This is the maximum-entropy solution under nonnegativity and (optional) prior.

    If prior is None, a uniform prior is used (independence model).
    If you have sampling counts per (lat, LST, time), pass them as prior (+ small epsilon).
    """
    nlat, nt1 = dens_lat.shape
    nlst, nt2 = dens_lst.shape
    if nt1 != nt2:
        raise ValueError("dens_lat and dens_lst must have the same number of time steps.")
    nt = nt1

    if w_lat is None:
        w_lat = np.ones(nlat, dtype=float)
    else:
        w_lat = np.asarray(w_lat, dtype=float)
    if w_lst is None:
        w_lst = np.ones(nlst, dtype=float)
    else:
        w_lst = np.asarray(w_lst, dtype=float)

    # Ensure nonnegative inputs; treat NaNs as zero (no constraint from that bin)
    dens_lat = np.nan_to_num(dens_lat, nan=0.0, copy=True)
    dens_lst = np.nan_to_num(dens_lst, nan=0.0, copy=True)
    dens_lat = np.clip(dens_lat, 0.0, None)
    dens_lst = np.clip(dens_lst, 0.0, None)

    X = np.zeros((nlat, nlst, nt), dtype=float)
    sum_w_lat = float(w_lat.sum())
    sum_w_lst = float(w_lst.sum())

    for t in range(nt):
        A = dens_lat[:, t].astype(float)    # lat-averages over LST
        B = dens_lst[:, t].astype(float)    # LST-averages over lat

        # Target row/column sums implied by averages and bin widths
        R = A * sum_w_lst            # shape (nlat,)
        C = B * sum_w_lat            # shape (nlst,)

        # Scale to make totals match (robust to small numerical differences)
        total_R = R.sum()
        total_C = C.sum()
        if total_R <= eps and total_C <= eps:
            # Nothing at this time slice
            continue
        elif total_R > eps and total_C > eps:
            C *= (total_R / (total_C + eps))
        elif total_R > eps and total_C <= eps:
            # No column mass but positive rows -> fall back to uniform columns
            C = np.full(nlst, total_R / nlst)
        else:
            # No row mass but positive columns -> fall back to uniform rows
            R = np.full(nlat, total_C / nlat)

        # Prior
        if prior is None:
            P = np.ones((nlat, nlst), dtype=float)
        else:
            P = prior[:, :, t].astype(float, copy=True)
            # Keep nonnegative and avoid exact zeros where mass is required
            P = np.clip(P, 0.0, None)
            if P.sum() == 0.0:
                P[:] = 1.0
            P += eps

        # Mask rows/cols with zero targets to avoid division by zero
        row_mask = R > eps
        col_mask = C > eps
        # If a row has R=0, force it to zero
        # If a column has C=0, force it to zero
        P_eff = P.copy()
        P_eff[~row_mask, :] = 0.0
        P_eff[:, ~col_mask] = 0.0
        P_eff += eps  # keep strictly positive where allowed

        # Sinkhorn iterations: find u, v such that X = diag(u) P diag(v) matches R, C
        u = np.ones(nlat, dtype=float)
        v = np.ones(nlst, dtype=float)
        for _ in range(max_iter):
            u_prev, v_prev = u, v

            Pv = P_eff @ v
            Pv = np.where(Pv > eps, Pv, 1.0)  # avoid 0 division
            u = np.where(row_mask, R / Pv, 0.0)

            PTu = P_eff.T @ u
            PTu = np.where(PTu > eps, PTu, 1.0)
            v = np.where(col_mask, C / PTu, 0.0)

            # Convergence check on multiplicative factors
            du = np.max(np.abs(u - u_prev) / (np.abs(u_prev) + eps))
            dv = np.max(np.abs(v - v_prev) / (np.abs(v_prev) + eps))
            if max(du, dv) < tol:
                break

        X[:, :, t] = (u[:, None] * P_eff) * v[None, :]

    return X

if __name__ == "__main__":
    main()
