import numpy as np
from scipy.sparse import diags, vstack, csr_matrix
from scipy.sparse.linalg import lsqr
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import pickle as pkl
import os
from tqdm import tqdm
import time
# make font arial


# === Laplacian for smoothing rho0 ===
def build_laplacian(n, weight):
    diagonals = [weight * np.ones(n - 1), -2 * weight * np.ones(n), weight * np.ones(n - 1)]
    return diags(diagonals, offsets=[-1, 0, 1], shape=(n, n))

# === Log-domain smoothing of H(t) with moving average ===
def smooth_log_H(H_vec, window=5):
    H_log = np.log(H_vec)
    H_log_smooth = uniform_filter1d(H_log, size=window, mode='nearest')
    return np.exp(H_log_smooth)

# === Main Function ===
def main():
    t1 = time.time()
    t_min = 5 # time resolution in minutes
    # === Load Data ===
    # file_path = 'density_history_june2015_v2_10m_filt.pkl'
    file_path = '../ml_decay_prediction/react_thesis/density_history_octG4_v2_5m_filt.pkl'
    file_path_save = 'data/octG4_20241006_' + str(t_min) + 'm_ds.pkl'
    with open(file_path, 'rb') as f:
        data = pkl.load(f)

    output_dir = file_path_save[:-4] + "_1D_owd"
    os.makedirs(output_dir, exist_ok=True)

    satcat_list = data['satcat']
    ds_list = np.array(data['ds'])
    time_list = data['time']
    lat_list = data['lat']
    lon_list = data['lon']
    tvec = data['tvec']
    h = np.array(data['alt'])
    m = len(ds_list)
    nt = len(tvec)

    # === Q matrix ===
    unique_satcats, sat_idx = np.unique(satcat_list, return_inverse=True)
    K = len(unique_satcats)
    Q = csr_matrix((np.ones(m), (np.arange(m), sat_idx)), shape=(m, K))

    # === Initialization ===
    d_s = ds_list
    rho0_vec = np.ones(nt) * 1e-10
    H_vec = np.ones(nt) * 100.0
    h0 = 600  # reference altitude
    n_iter = 5
    L_rho = build_laplacian(nt, weight=1500)

    # === Stage 1: Build A_global matrix ===
    A_global = np.zeros((m, nt))
    for i in tqdm(range(m), desc="Building Global A"):
        t_bins = np.searchsorted(tvec, time_list[i])
        t_bins = t_bins[(t_bins >= 0) & (t_bins < nt)]
        A_global[i, t_bins] = 1.0 / len(t_bins) if len(t_bins) > 0 else 0

    loss_history = []

    # count the number of measurements in each time bin
    counts = np.sum(A_global > 0, axis=0)
    valid = counts > np.average(counts) * 0.25

    # plt.figure()
    # plt.plot(tvec, counts, marker='o', linestyle='-')
    # plt.title("Measurements per Time Bin")
    # plt.xlabel("Time")
    # plt.show()

    # === Coordinate Descent Loop ===
    for iteration in range(n_iter):
        print(f"\n--- Iteration {iteration} ---")

        # Step 1: Interpolate model values
        rho_interp = A_global @ rho0_vec
        H_interp = A_global @ H_vec

        # Step 2: Solve for beta
        rho_eff = rho_interp * np.exp(-(h - h0) / H_interp)
        y_beta = d_s / (rho_eff + 1e-14)
        beta_vec = lsqr(Q, y_beta)[0]

        # Normalize beta per altitude bin
        alt_bins = np.linspace(np.min(h), np.max(h), 10)
        sat_avg_alt = np.bincount(sat_idx, weights=h) / np.bincount(sat_idx)
        alt_idx = np.digitize(sat_avg_alt, alt_bins) - 1
        for i in range(len(alt_bins) - 1):
            bin_mask = (alt_idx == i)
            if np.any(bin_mask):
                beta_vec[bin_mask] /= np.mean(beta_vec[bin_mask])

        # beta_vec = beta_vec / np.mean(beta_vec)  # Normalize overall

        # Step 3: Solve for rho0 with Laplacian smoothing
        beta_eff = beta_vec[sat_idx]
        expo = np.exp(-(h - h0) / H_interp)
        rhs_rho = d_s / (beta_eff * expo + 1e-14)
        A_reg = vstack([csr_matrix(A_global), L_rho])
        b_reg = np.concatenate([rhs_rho, np.zeros(nt)])
        rho0_vec = lsqr(A_reg, b_reg)[0]
        # rho0_vec = np.clip(rho0_vec, 1e-15, None)
        rho0_vec = np.clip(rho0_vec, 0.05 * np.max(rho0_vec), None)


        # Step 4: Solve for H(t)
        H_vec_new = H_vec.copy()
        for t in range(nt):
            w = A_global[:, t]
            idx_t = np.where(w > 0)[0]
            if len(idx_t) < 5:
                continue
            weights = w[idx_t]
            y = np.log(d_s[idx_t] / (beta_vec[sat_idx[idx_t]] * rho0_vec[t] + 1e-14))
            x = h[idx_t] - h0
            A = np.vstack([x, np.ones_like(x)]).T
            try:
                W = np.diag(weights)
                AtW = A.T @ W
                slope, _ = np.linalg.lstsq(AtW @ A, AtW @ y, rcond=None)[0]
                if slope < 0:
                    H_vec_new[t] = -1 / slope
            except:
                pass

        # Step 5: Smooth H in log domain
        H_vec = smooth_log_H(H_vec_new, window=20)
        H_vec = np.clip(H_vec, 1.0, 1000.0)

        # Step 6: Compute residuals and loss
        rho_interp = A_global @ rho0_vec
        H_interp = A_global @ H_vec
        pred = beta_vec[sat_idx] * rho_interp * np.exp(-(h - h0) / H_interp)
        residuals = np.log(d_s + 1e-14) - np.log(pred + 1e-14)
        log_loss = np.sum(residuals ** 2)
        loss_history.append(log_loss)
        print(f"Log Loss = {log_loss:.3e}")
        if loss_history[-1] > (loss_history[-2] if len(loss_history) > 1 else np.inf):
            print("Warning: Loss increased, optimization may not be converging.")
            break
    
    print(f"Total time taken: {time.time() - t1:.2f} seconds")

    # === Save results ===
    results = {
        'tvec': tvec,
        'rho0_vec': rho0_vec,
        'H_vec': H_vec,
        'loss_history': loss_history,
        'beta_vec': beta_vec,
        'counts': counts,
    }
    with open(f"{output_dir}/results.pkl", 'wb') as f:
        pkl.dump(results, f)

    # === Plot results ===
    plt.figure(figsize=(4, 4))
    plt.grid(True, color='gray', linewidth=0.25)
    plt.plot(tvec[valid], H_vec[valid])
    plt.title("Scale Height H")
    plt.xlabel("Date (UTC)")
    plt.ylabel("H [km]")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scale_height.png")

    plt.figure(figsize=(4, 4))
    plt.grid(True, color='gray', linewidth=0.25)
    plt.plot(tvec[valid], rho0_vec[valid])
    plt.title(r"BC-Scaled $\rho_0$ at 600 km")
    plt.xlabel("Date (UTC)")
    plt.ylabel(r"$\rho_0$ [kg/m³]")
    # plt.yscale('log')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rho0.png")

    # === Compute density for h_query ===
    h_query_vec = np.array([300,600,900])#np.linspace(200, 1000, 10)
    beta_avg = 1e-7
    dens_vec = (1 / beta_avg) * rho0_vec * np.exp(-(h_query_vec[:, None] - h0) / H_vec[None, :]) / 1e9

    plt.figure(figsize=(7, 4))
    plt.grid(True, color='lightgray', linewidth=0.25)
    for i, h_val in enumerate(h_query_vec):
        plt.semilogy(tvec[valid], dens_vec[i, valid], label=f"h = {h_val:.0f} km")
    plt.legend()
    plt.xticks(rotation=45)
    plt.xlabel('Date (UTC)')
    plt.ylabel('Neutral Density [kg/m³]')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/density_profiles_all_altitudes.png")

    h_query = 500
    dens = (1 / beta_avg) * rho0_vec * np.exp(-(h_query - h0) / H_vec) / 1e9
    dens_smooth = np.convolve(dens, np.ones(10) / 10, mode='same')

    plt.figure(figsize=(7, 4))
    plt.grid(True, color='gray', linewidth=0.25)
    plt.plot(tvec[valid], dens_smooth[valid])
    plt.xlabel("Date (UTC)")
    plt.ylabel("Neutral Density [kg/m³]")
    plt.xticks(rotation=45)
    plt.title(f"Density at {h_query} km")
    plt.tight_layout()
    # plt.savefig(f"{output_dir}/density_500km_coord_descent.png")
    plt.show()



if __name__ == "__main__":
    main()
