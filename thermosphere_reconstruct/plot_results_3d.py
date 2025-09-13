# load results and plot
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os
import spacepy.pycdf as pycdf

def main():
    # open pkl file with results
    # === Load results from pkl file
    t_min = 10
    # file_path = 'data/gannon_20240509_' + str(t_min) + 'm_ds.pkl'
    # file_path = 'data/octG4_20241006_' + str(t_min) + 'm_ds.pkl'
    file_path = 'data/gannon_20240509_' + str(t_min) + 'm_ds.pkl'
    output_dir = file_path[:-4] + "_3D_owd"
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist.")

    with open(f"{output_dir}/results.pkl", 'rb') as f:
        results = pkl.load(f)
    tvec = results['tvec']
    rho0_vec = results['rho0_vec']
    H_vec = results['H_vec']
    beta_vec = results['beta_vec']
    counts = results['counts']
    h0 = 600
    lat_bins = np.linspace(-90, 90, rho0_vec.shape[1] + 1)  # assuming lat_bins are in results
    lst_bins = np.linspace(0,24, rho0_vec.shape[2] + 1)  # assuming lst_bins are in results

    # pick 100 indices to plot
    plt_idx = np.linspace(0, rho0_vec.shape[0] - 1, 100).astype(int)

    # for each timestep, plot rho0 and H
    maxrho0 = np.nanmax(rho0_vec[:])
    maxH = np.nanmax(H_vec[:])
    for i in plt_idx:
        plt.figure(figsize=(12, 3))

        plt.subplot(1, 2, 1)
        plt.imshow(rho0_vec[i], aspect='auto', origin='lower', extent=[lst_bins[0], lst_bins[-1], lat_bins[0], lat_bins[-1]], cmap='jet')#, interpolation='gaussian')
        plt.colorbar(label='Density (kg/m³)')
        # plt.clim(0, maxrho0)
        plt.title(f'BC-scaled rho0 at {tvec[i]}')
        plt.xlabel('Date (UTC)')
        plt.ylabel('Latitude (°)')
        
        plt.subplot(1, 2, 2)
        plt.imshow(H_vec[i], aspect='auto', origin='lower', extent=[lst_bins[0], lst_bins[-1], lat_bins[0], lat_bins[-1]], cmap='jet')#, interpolation='gaussian')
        plt.colorbar(label='Scale Height (km)')
        # plt.clim(0, maxH)
        plt.title(f'Scale Height at {tvec[i]}')
        plt.xlabel('Date (UTC)')
        plt.ylabel('Latitude (°)')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rho0_H_{i:03d}.png")
        plt.show()
        print(i/rho0_vec.shape[0], end='\r')
    # plot
    c = 3

if __name__ == "__main__":
    main()
