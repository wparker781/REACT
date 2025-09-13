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

# === Laplacian for smoothing rho0 ===
def build_laplacian(n, weight):
    diagonals = [weight * np.ones(n - 1), -2 * weight * np.ones(n), weight * np.ones(n - 1)]
    return diags(diagonals, offsets=[-1, 0, 1], shape=(n, n))

def build_laplacian_2d(nt, nlst, weight_t=1.0, weight_lst=1.0):
    L_t = build_laplacian(nt, weight=weight_t)
    L_lst = build_laplacian(nlst, weight=weight_lst)
    # L_2d = np.kron(np.eye(nlst), L_t.toarray()) + np.kron(L_lst.toarray(), np.eye(nt))
    L_2d = np.kron(np.eye(nt), L_lst.toarray()) + np.kron(L_t.toarray(), np.eye(nlst))

    return csr_matrix(L_2d)

def smooth_H_box(H_mat, size=(5, 3)):
    """
    Apply a 2D moving average to smooth H(t, lst)
    
    Parameters:
    - H_mat: array of shape (nt, nlst)
    - size: smoothing window in (time, lst) dimensions

    Returns:
    - Smoothed H_mat of same shape
    """
    return np.exp(uniform_filter(np.log(H_mat), size=size, mode='nearest'))

# === Log-domain smoothing of H(t, lst) with moving average ===
def smooth_log_H(H_mat, window=5):
    H_log = np.log(H_mat)
    H_log_smooth = uniform_filter1d(H_log, size=window, axis=0, mode='nearest')
    return np.exp(H_log_smooth)

# === Main Function ===
def main():
    t0 = time.time()
    t_min = 5
    # === Load Data ===
    file_path = '../ml_decay_prediction/react_thesis/density_history_octG4_v2_5m_filt.pkl'
    file_path_save = 'data/octG4_20241006_' + str(t_min) + 'm_ds.pkl'
    with open(file_path, 'rb') as f:
        data = pkl.load(f)

    output_dir = file_path_save[:-4] + "_2D_lst_owd"
    os.makedirs(output_dir, exist_ok=True)

    satcat_list = data['satcat']
    ds_list = np.array(data['ds'])
    h = np.array(data['alt'])
    time_list = data['time']  # list of lists
    # lst_list = data['lst']    # list of lists
    lon_list = data['lon']    # list of lists
    tvec = data['tvec']
    # make a new tvec that is every 5th element
    tvec = tvec[::5]
    m = len(ds_list)
    nt = len(tvec)

    # convert to LST
    lst_list = []
    for i in range(m):
        dt_index = pd.DatetimeIndex(time_list[i])
        utc_hours = dt_index.hour + dt_index.minute / 60.0 + dt_index.second / 3600.0
        lst_list.append((lon_list[i] / 15.0 + utc_hours) % 24)

    lst_bins = np.linspace(0, 24, 10) # call this lst for now, but its really lst!
    nlst = len(lst_bins) - 1

    # === Flatten and build full measurement array ===
    d_s_all, h_all, time_idx_all, lst_idx_all, sat_idx_all = [], [], [], [], []

    unique_satcats, satcat_indices = np.unique(satcat_list, return_inverse=True)
    n_satcat = len(unique_satcats)

    # compute time indices and lst indices for each measurment in time_list and lst_list
    time_idx = []
    lst_idx = []
    for i in range(m):
        time_idx.append(np.searchsorted(tvec, time_list[i]))
        lst_idx.append(np.digitize(lst_list[i], lst_bins) - 1)

    # for i in range(m):
    #     d_i = ds_list[i]
    #     h_i = h[i]
    #     s_i = satcat_indices[i]

    #     for t, lst in zip(time_list[i], lst_list[i]):
    #         t_bin = np.searchsorted(tvec, t)
    #         if t_bin < 0 or t_bin >= nt:
    #             continue
    #         lst_bin = np.digitize(lst, lst_bins) - 1
    #         if lst_bin < 0 or lst_bin >= nlst:
    #             continue

    #         d_s_all.append(d_i)
    #         h_all.append(h_i)
    #         time_idx_all.append(t_bin)
    #         lst_idx_all.append(lst_bin)
    #         sat_idx_all.append(s_i)

    # d_s_all = np.array(d_s_all)
    # h_all = np.array(h_all)
    # time_idx_all = np.array(time_idx_all)
    # lst_idx_all = np.array(lst_idx_all)
    # sat_idx_all = np.array(sat_idx_all)
    # m_expanded = len(d_s_all)

    # === Q matrix ===
    Q = csr_matrix((np.ones(m), (np.arange(m), satcat_indices)), shape=(m, n_satcat))

    # === Stage 1: Build A_global matrix ===
    A_global = np.zeros((m, nt*nlst))
    counts = np.zeros((nt, nlst))
    for i in tqdm(range(m), desc="Building Global A"):
        t_bins = time_idx[i]
        idx_valid = (t_bins >= 0) & (t_bins < nt)
        t_bins = t_bins[idx_valid]
        lst_bins_i = lst_idx[i]
        lst_bins_i = lst_bins_i[idx_valid]
        counts[t_bins, lst_bins_i] += 1
        A_global[i, t_bins * nlst + lst_bins_i] = 1.0 / len(t_bins) if len(t_bins) > 0 else 0

    # convert A_global to sparse matrix
    A_global = csr_matrix(A_global)

    # === A_global: measurement-to-(t,lst) bin mapping ===
    # A_global = csr_matrix((np.ones(m), (np.arange(m), time_idx_all * nlst + lst_idx_all)), shape=(m, nt * nlst))
    # A_global = csr_matrix((np.ones(m_expanded), (np.arange(m_expanded), lst_idx_all * nlst + time_idx_all)), shape=(m_expanded, nt * nlst))
    
    # === Initialization ===
    h0 = 600
    n_iter = 3
    rho0_vec = np.ones(nt * nlst) * 1e-10
    H_vec = np.ones(nt * nlst) * 100.0

    # L_t = build_laplacian(nt, weight=100).toarray()
    # L_lst = build_laplacian(nlst, weight=10).toarray()
    # L_H = build_laplacian_2d(nt, nlst, weight_t=100, weight_lst=1)
    # A_H = vstack([np.eye(nt * nlst), L_H])

    # L_rho = csr_matrix(np.kron(np.eye(nlst), L_t) + np.kron(L_lst, np.eye(nt)))
    L_rho = build_laplacian_2d(nt, nlst, weight_t=50, weight_lst=0.5)  # adjust weights as needed
    # rewrite without csr_matrix

    loss_history = []

    rho_interp = A_global @ rho0_vec
    H_interp = A_global @ H_vec

    # === Coordinate Descent ===
    for iteration in range(n_iter):
        print(f"\n--- Iteration {iteration} ---")

        rho_eff = rho_interp * np.exp(-(h - h0) / H_interp)
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
        expo = np.exp(-(h - h0) / H_interp)
        rhs_rho = ds_list / (beta_eff * expo + 1e-16)

        A_reg = vstack([A_global, L_rho])
        b_reg = np.concatenate([rhs_rho, np.zeros(L_rho.shape[0])])

        rho0_vec = lsqr(A_reg, b_reg)[0]
        # set min clip value to be 5% of the max rho0_vec
        rho0_vec = np.clip(rho0_vec, 0.05 * np.max(rho0_vec), None)

        # Step 4: Estimate H (per (t,lst) bin)
        H_vec_new = H_vec.copy()
        for idx in range(nt * nlst):
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
                    H_vec_new[idx] = -1 / slope
                
            except:
                pass
        
        #         A_reg = vstack([A_global, L_rho])
        # b_reg = np.concatenate([rhs_rho, np.zeros(L_rho.shape[0])])

        # rho0_vec = lsqr(A_reg, b_reg)[0]

        H_vec = smooth_H_box(H_vec_new.reshape(nt, nlst), size=(20, 5)).reshape(nt * nlst) # 20,5

        # H_vec = H_vec_new.copy()
        # b_H = np.concatenate([H_vec_new, np.zeros(L_H.shape[0])])
        # H_vec = lsqr(A_H, b_H)[0]
        # H_vec = smooth_log_H(H_vec_new.reshape(nlst, nt), window=10).reshape(nt * nlst)
        # H_vec = smooth_log_H(H_vec_new.reshape(nt, nlst), window=10).reshape(nt * nlst)
        # H_vec = H_vec_new
        # H_vec = np.clip(H_vec, 1.0, 1000.0)

        rho_interp = A_global @ rho0_vec
        H_interp = A_global @ H_vec
        pred = beta_eff * rho_interp * np.exp(-(h - h0) / H_interp)
        residuals = np.log(ds_list + 1e-16) - np.log(pred + 1e-16)
        log_loss = np.sum(residuals ** 2)
        loss_history.append(log_loss)
        print(f"Log Loss = {log_loss:.3e}")

    print(f"Total time: {time.time() - t0:.2f} seconds")

    # === Save results ===
    results = {
        'tvec': tvec,
        'lst_bins': lst_bins,
        'rho0_vec': rho0_vec.reshape(nt, nlst),
        'H_vec': H_vec.reshape(nt, nlst),
        'loss_history': loss_history,
        'beta_vec': beta_vec,
        'counts': counts,
        'satcat_list': satcat_list
    }

    with open(f"{output_dir}/results.pkl", 'wb') as f:
        pkl.dump(results, f)

    # # figure out the counts in each lst/time bin. If the count is < 10, set it to nan
    # counts = np.zeros((nlst, nt))
    # for i in range(m):
    #     t_bins = time_idx[i]
    #     lst_bins_i = lst_idx[i]
    #     counts[lst_bins_i, t_bins] += 1
    
    # set results['rho0_vec'][counts < 10] = np.nan
    counts_avg = np.average(counts)
    results['rho0_vec'][counts < counts_avg*0.25] = np.nan
    results['H_vec'][counts < counts_avg*0.25] = np.nan

    # plot rho0, H, and beta
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(results['rho0_vec'].T, aspect='auto', origin='lower', extent=[tvec[0], tvec[-1], lst_bins[0], lst_bins[-1]], cmap='jet')
    # plt.imshow(results['rho0_vec'].T, aspect='auto', extent=[lat_bins[0], lat_bins[-1], tvec[0], tvec[-1]], origin='lower')
    plt.colorbar(label='Density (kg/m³)')
    plt.title('BC-scaled rho0')
    # plt.xlabel('Date (UTC)')
    plt.ylabel('LST (hours)')
    plt.subplot(2, 1, 2)
    plt.imshow(results['H_vec'].T, aspect='auto', origin='lower', extent=[tvec[0], tvec[-1], lst_bins[0], lat_bins[-1]], cmap='jet')
    plt.colorbar(label='Scale Height (km)')
    plt.title('Scale Height')
    # plt.xlabel('Date (UTC)')
    plt.ylabel('LST (hours)')
    # plt.subplot(3, 1, 3)
    # plt.hist(results['beta_vec'])
    # plt.title('Estimated Beta')
    # plt.xlabel('Beta')
    # plt.ylabel('Frequency')
    # plt.tight_layout()
    plt.show()

    # make a plot of the combined density considering beta, rho0, and H
    beta_avg = 1e-7
    alt_query = 500
    dens_2d = (1/beta_avg)* results['rho0_vec'].T  * np.exp(-(alt_query-h0) / results['H_vec'].T) / 1e9 # convert rho0 to kg/m³
    plt.figure(figsize=(12, 6))
    plt.imshow(dens_2d, aspect='auto', origin='lower', extent=[tvec[0], tvec[-1], lst_bins[0], lst_bins[-1]], cmap='jet')
    plt.colorbar(label='Density (kg/m³)')
    plt.title(f'Neutral Density at {alt_query} km')
    plt.xlabel('Date (UTC)')
    plt.ylabel('LST (hours)')
    plt.show()

    # plot a 3x3 grid of density profiles at different altitudes from 400 to 1000 km
    h_query_vec = np.linspace(400, 1000, 9)
    plt.figure(figsize=(12, 8))
    for i, h_query in enumerate(h_query_vec):
        dens_query = (1/beta_avg)* results['rho0_vec'].T  * np.exp(-(h_query-h0) / results['H_vec'].T) / 1e9 # convert rho0 to kg/m³
        plt.subplot(3, 3, i + 1)
        plt.imshow(dens_query, aspect='auto', origin='lower', extent=[tvec[0], tvec[-1], lst_bins[0], lst_bins[-1]], cmap='jet')
        # don't show x axis labels
        plt.xticks([])
        plt.colorbar(label='Density (kg/m³)')
        plt.title(f'Neutral Density at {h_query:.0f} km')
        plt.ylabel('LST (hours)')
    plt.tight_layout()
    # plt.savefig(f"{output_dir}/density_profiles_all_altitudes.png")
    plt.show()

                                                                 


if __name__ == "__main__":
    main()
