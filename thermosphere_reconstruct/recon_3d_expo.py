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
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: There appear to be .* leaked semaphore objects"
)

# === Main Function ===
def main():
    t0 = time.time()
    # === Load Data ===
    t_min = 10
    # file_path = 'density_history_gannon_v2_' + str(t_min) + 'm_filt.pkl'
    # file_path = 'test_20250101_' + str(t_min) + 'm_ds.pkl'
    file_path = '../ml_decay_prediction/react_thesis/density_history_gannon_v2_10m_filt.pkl'
    file_path_save = 'data/gannon_20240509_' + str(t_min) + 'm_ds.pkl'    
    with open(file_path, 'rb') as f:
        data = pkl.load(f)

    # output_dir = "output_lat_lst_time_fit"
    output_dir = file_path_save[:-4] + "_3D_owd"
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
    # tvec = tvec[::3]
    m = len(ds_list)
    nt = len(tvec)

    print(np.min(ds_list), np.max(ds_list))

    # plt.figure()
    # plt.semilogy(ds_list)
    # plt.show()


    # === Latitude binning ===
    lat_bins = np.linspace(-90, 90, 5)
    nlat = len(lat_bins) - 1
        # convert to LST
    lst_list = []
    for i in range(m):
        dt_index = pd.DatetimeIndex(time_list[i])
        utc_hours = dt_index.hour + dt_index.minute / 60.0 + dt_index.second / 3600.0
        lst_list.append((lon_list[i] / 15.0 + utc_hours) % 24)

    lst_bins = np.linspace(0, 24, 10)
    nlst = len(lst_bins) - 1

    # === Flatten and build full measurement array ===
    # d_s_all, h_all, time_idx_all, lat_idx_all, sat_idx_all = [], [], [], [], []

    unique_satcats, satcat_indices = np.unique(satcat_list, return_inverse=True)
    n_satcat = len(unique_satcats)

    # compute time indices and lat indices for each measurment in time_list and lat_list
    time_idx = []
    lat_idx = []
    lst_idx = []
    for i in range(m):
        time_idx.append(np.searchsorted(tvec, time_list[i]))
        lat_idx.append(np.digitize(lat_list[i], lat_bins) - 1)
        lst_idx.append(np.digitize(lst_list[i], lst_bins) - 1)

    # === Q matrix ===
    Q = csr_matrix((np.ones(m), (np.arange(m), satcat_indices)), shape=(m, n_satcat))

    # === Stage 1: Build A_global matrix ===
    A_global = np.zeros((m, nt*nlat*nlst))
    counts = np.zeros((nt, nlat, nlst))
    weights_per_obs = np.zeros(m)

    # for i in tqdm(range(m), desc="Building Global A"):
    #     t_bins = time_idx[i]
    #     idx_valid = (t_bins >= 0) & (t_bins < nt)
    #     t_bins = t_bins[idx_valid]
    #     lat_bins_i = lat_idx[i]
    #     lat_bins_i = lat_bins_i[idx_valid]
    #     lst_bins_i = lst_idx[i]
    #     lst_bins_i = lst_bins_i[idx_valid]
    #     counts[t_bins, lat_bins_i, lst_bins_i] += 1

    #     # t_bins * (nlat * nlst) + lat_bins_i * nlst + lst_bins_i
    #     A_global[i, t_bins * (nlat * nlst) + lat_bins_i * nlst + lst_bins_i] = 1.0 / len(t_bins) if len(t_bins) > 0 else 0 # CHECK THIS!!!


    rows, cols, vals = [], [], []
    for i in range(m):
        t_bins = time_idx[i]
        valid = (t_bins >= 0) & (t_bins < nt)
        t_bins = t_bins[valid]
        lat_bins_i = lat_idx[i][valid]
        lst_bins_i = lst_idx[i][valid]
        if len(t_bins) == 0:
            continue
        col_idx = t_bins * (nlat * nlst) + lat_bins_i * nlst + lst_bins_i
        rows.extend([i] * len(col_idx))
        cols.extend(col_idx.tolist())
        vals.extend([1.0 / len(col_idx)] * len(col_idx))

    A_global = csr_matrix((vals, (rows, cols)), shape=(m, nt * nlat * nlst))

    # === Compute weights (inverse counts) ===
    weights_per_obs = np.zeros(m)
    for i in range(m):
        t_bins = np.searchsorted(tvec, time_list[i])
        lat_bins_i = np.digitize(lat_list[i], lat_bins) - 1
        lst_bins_i = np.digitize(lst_list[i], lst_bins) - 1
        valid = (t_bins >= 0) & (t_bins < nt)
        if np.any(valid):
            bin_counts = counts[t_bins[valid], lat_bins_i[valid], lst_bins_i[valid]]
            weights_per_obs[i] = np.mean(1.0 / (bin_counts + 1e-6))

    # convert A_global to sparse matrix
    A_global = csr_matrix(A_global)

    # === Initialization ===
    h0 = 600
    n_iter = 3
    rho0_vec = np.ones(nt * nlat * nlst) * 1e-10
    H_vec = np.ones(nt * nlat * nlst) * 100.0

    # L_rho = build_laplacian_2d(nt, nlat, weight_t=200, weight_lat=0.5)  # adjust weights as needed
    L_rho = build_laplacian_3d(nt, nlat, nlst, weight_t=500, weight_lat=5, weight_lst=5)  # 1, 0.5, 0.5 # adjust weights as needed

    loss_history = []

    rho_interp = A_global @ rho0_vec
    H_interp = A_global @ H_vec

    # === Coordinate Descent ===
    for iteration in range(n_iter):
        print(f"\n--- Iteration {iteration} ---")

        denom = np.clip(H_interp, 10.0, 800.0)  # physical range of scale heights (km)
        rho_eff = rho_interp * np.exp(-(h - h0) / denom)
        # rho_eff = rho_interp * np.exp(-(h - h0) / (H_interp+1e-6))
        y_beta = ds_list / (rho_eff + 1e-16)
        beta_vec = lsqr(Q, y_beta)[0]

        # Normalize beta per altitude bin
        alt_bins = np.linspace(np.min(h), np.max(h), 10)
        sat_avg_alt = np.bincount(satcat_indices, weights=h) / np.bincount(satcat_indices)
        alt_idx = np.digitize(sat_avg_alt, alt_bins) - 1
        for i in range(len(alt_bins) - 1):
            bin_mask = (alt_idx == i)
            if np.any(bin_mask):
                beta_vec[bin_mask] /= np.mean(beta_vec[bin_mask])

        beta_eff = beta_vec[satcat_indices]
        # expo = np.exp(-(h - h0) / (H_interp + 1e-6))
        expo = np.exp(-(h - h0) / denom)
        rhs_rho = ds_list / (beta_eff * expo + 1e-16)

        if weighting == True:
            W_sqrt = diags(np.sqrt(weights_per_obs))
            A_weighted = W_sqrt @ A_global
            b_weighted = weights_per_obs * rhs_rho

            A_reg = vstack([A_weighted, L_rho])
            b_reg = np.concatenate([b_weighted, np.zeros(L_rho.shape[0])])
        else:
            A_reg = vstack([A_global, L_rho])
            b_reg = np.concatenate([rhs_rho, np.zeros(L_rho.shape[0])])

        rho0_vec = lsqr(A_reg, b_reg)[0]
        # set min clip value to be 5% of the max rho0_vec
        rho0_vec = np.clip(rho0_vec, 0.05 * np.max(rho0_vec), None)

        # Step 4: Estimate H (per (t,lat) bin)
        H_vec_new = H_vec.copy()
        for idx in range(nt * nlat * nlst):
            idx_mask = A_global[:, idx].toarray().flatten() > 0
            if np.sum(idx_mask) < 5:
                continue
            weights = A_global[idx_mask, idx].toarray().flatten()
            y = np.log(ds_list[idx_mask] / (beta_eff[idx_mask] * rho0_vec[idx] + 1e-16))
            x = h[idx_mask] - h0
            A = np.vstack([x, np.ones_like(x)]).T
            try:
                W = np.diag(weights)
                AtW = A.T @ W
                slope, _ = np.linalg.lstsq(AtW @ A, AtW @ y, rcond=None)[0]
                if slope < 0:
                    if slope < -1e-6:
                        H_vec_new[idx] = np.clip(-1 / slope, 10.0, 800.0)
                    else:
                        H_vec_new[idx] = 100.0  # fallback scale height
                
            except:
                pass

        H_vec = smooth_H_box_3d(H_vec_new.reshape(nt, nlat, nlst), size=(5, 5, 3)).reshape(nt * nlat * nlst)
        H_vec = np.clip(H_vec, 10.0, 800.0)

        # plt.figure()
        # plt.subplot(2,1,1)
        # plt.plot(H_vec)
        # plt.title('Estimated Scale Height H(t, lat, lst)')
        # plt.subplot(2,1,2)
        # plt.semilogy(rho0_vec)
        # plt.title('Estimated Density rho(t, lat, lst)')
        # plt.show()

        rho_interp = A_global @ rho0_vec
        H_interp = A_global @ H_vec

        # plt.figure()
        # plt.subplot(3, 1, 1)
        # plt.semilogy(ds_list)
        # plt.title('Observed Density ds(t, lat, lst)')
        # plt.subplot(3, 1, 2)
        # plt.plot(H_interp)
        # plt.title('Estimated Scale Height H(t, lat, lst)')
        # plt.subplot(3, 1, 3)
        # plt.plot(rho_interp)
        # plt.title('Estimated Density rho(t, lat, lst)')
        # plt.show()

        pred = beta_eff * rho_interp * np.exp(-(h - h0) / (H_interp))
        residuals = np.log(ds_list + 1e-16) - np.log(pred + 1e-16)
        log_loss = np.sum(residuals ** 2)
        loss_history.append(log_loss)
        print(f"Log Loss = {log_loss:.3e}")

    print(f"Total time: {time.time() - t0:.2f} seconds")

    plt.figure()
    plt.plot(rho0_vec)
    plt.title('rho0(t, lat, lst)')
    plt.show()

    # === Save results ===
    results = {
        'tvec': tvec,
        'lat_bins': lat_bins,
        'lst_bins': lst_bins,
        'rho0_vec': rho0_vec.reshape(nt, nlat, nlst),
        'H_vec': H_vec.reshape(nt, nlat, nlst),
        'loss_history': loss_history,
        'beta_vec': beta_vec,
        'counts': counts,
    }

    with open(f"{output_dir}/results.pkl", 'wb') as f:
        pkl.dump(results, f)
    
    # set results['rho0_vec'][counts < 10] = np.nan
    # results['rho0_vec'][counts < 10] = np.nan
    # results['H_vec'][counts < 10] = np.nan

    # for each timestep, average rho0 and H over lat and lst
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


# === Laplacian for smoothing rho0 ===
def build_laplacian(n, weight):
    diagonals = [weight * np.ones(n - 1), -2 * weight * np.ones(n), weight * np.ones(n - 1)]
    return diags(diagonals, offsets=[-1, 0, 1], shape=(n, n))

def build_laplacian_2d(nt, nlat, weight_t=1.0, weight_lat=1.0):
    L_t = build_laplacian(nt, weight=weight_t)
    L_lat = build_laplacian(nlat, weight=weight_lat)
    # L_2d = np.kron(np.eye(nlat), L_t.toarray()) + np.kron(L_lat.toarray(), np.eye(nt))
    L_2d = np.kron(np.eye(nt), L_lat.toarray()) + np.kron(L_t.toarray(), np.eye(nlat))

    return csr_matrix(L_2d)

def build_laplacian_3d(nt, nlat, nlst, weight_t=1.0, weight_lat=1.0, weight_lst=1.0):
    L_t = build_laplacian(nt, weight=weight_t)
    L_lat = build_laplacian(nlat, weight=weight_lat)
    L_lst = build_laplacian(nlst, weight=weight_lst)
    
    # 3D Laplacian using Kronecker product
    L_3d = (
    np.kron(np.eye(nt * nlat), L_lst.toarray()) +
    np.kron(np.eye(nt), np.kron(L_lat.toarray(), np.eye(nlst))) +
    np.kron(L_t.toarray(), np.eye(nlat * nlst))
    )
    # L_3d = np.kron(np.eye(nt * nlat), L_lst.toarray()) + np.kron(L_t.toarray(), np.eye(nlat * nlst)) + np.kron(np.eye(nt), L_lat.toarray())

    return csr_matrix(L_3d)

def smooth_H_box(H_mat, size=(5, 3)):
    """
    Apply a 2D moving average to smooth H(t, lat)
    
    Parameters:
    - H_mat: array of shape (nt, nlat)
    - size: smoothing window in (time, latitude) dimensions

    Returns:
    - Smoothed H_mat of same shape
    """
    return np.exp(uniform_filter(np.log(H_mat), size=size, mode='nearest'))

def smooth_H_box_3d(H_mat, size=(5, 3, 3)):
    """
    Apply a 3D moving average to smooth H(t, lat, lst)
    
    Parameters:
    - H_mat: array of shape (nt, nlat, nlst)
    - size: smoothing window in (time, latitude, lst) dimensions

    Returns:
    - Smoothed H_mat of same shape
    """
    return np.exp(uniform_filter(np.log(H_mat), size=size, mode='nearest'))

# === Log-domain smoothing of H(t, lat) with moving average ===
def smooth_log_H(H_mat, window=5):
    H_log = np.log(H_mat)
    H_log_smooth = uniform_filter1d(H_log, size=window, axis=0, mode='nearest')
    return np.exp(H_log_smooth)

                                                                 
if __name__ == "__main__":
    main()
