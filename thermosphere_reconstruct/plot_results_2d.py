# load results and plot
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os
import spacepy.pycdf as pycdf
import matplotlib.gridspec as gridspec

def main():
    # open pkl file with results
    # === Load results from pkl file
    t_min = 5
    # file_path = 'data/gannon_20240509_' + str(t_min) + 'm_ds.pkl'
    # file_path = 'data/octG4_20241006_' + str(t_min) + 'm_ds.pkl'
    file_path = 'data/gannon_20240509_' + str(t_min) + 'm_ds.pkl'
    output_dir = file_path[:-4] + "_2D_lst_owd"
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
    # lat_bins = np.linspace(-90, 90, rho0_vec.shape[1] + 1)[:-1]  # assuming lat_bins are in results
    lat_bins = np.linspace(0,24, rho0_vec.shape[0] + 1)[:-1]  # assuming lst_bins are in results

    # plot

    # set results['rho0_vec'][counts < 10] = np.nan
    counts_avg = np.average(counts)
    results['rho0_vec'][counts < counts_avg*0.25] = np.nan
    results['H_vec'][counts < counts_avg*0.25] = np.nan



    plt.figure(figsize=(5, 8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 2])

    # Top: rho0
    ax1 = plt.subplot(gs[0])
    im1 = ax1.imshow(results['rho0_vec'].T, aspect='auto', origin='lower',
                    extent=[tvec[0], tvec[-1], lat_bins[0], lat_bins[-1]],
                    cmap='jet', interpolation='gaussian')
    # share xticks with the next plot
    # ax1.set_xticks([])
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.colorbar(im1, ax=ax1, label='Density (kg/m³)')
    ax1.set_title(r'BC-scaled $\rho_0$')
    # ax1.set_ylabel('Latitude (°)')
    ax1.set_ylabel('LST (hours)')
    # add grid
    ax1.grid(True, color='white', linewidth=0.16)

    # Middle: H
    ax2 = plt.subplot(gs[1])
    im2 = ax2.imshow(results['H_vec'].T, aspect='auto', origin='lower',
                    extent=[tvec[0], tvec[-1], lat_bins[0], lat_bins[-1]],
                    cmap='jet', interpolation='gaussian')
    plt.colorbar(im2, ax=ax2, label='Scale Height (km)')
    # ax2.set_xticks([])
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set_title('Scale Height, H')
    # ax2.set_ylabel('Latitude (°)')
    ax2.set_ylabel('LST (hours)')
    ax2.grid(True, color='white', linewidth=0.16)



    # Bottom: Density at alt_query
    beta_avg = 3.5e-7#8e-8
    alt_query = 500
    dens_2d = (1/beta_avg) * results['rho0_vec'].T * np.exp(-(alt_query-h0) / results['H_vec'].T) / 1e9
    ax3 = plt.subplot(gs[2])
    im3 = ax3.imshow(dens_2d, aspect='auto', origin='lower',
                    extent=[tvec[0], tvec[-1], lat_bins[0], lat_bins[-1]],
                    cmap='jet', interpolation='gaussian')
    plt.colorbar(im3, ax=ax3, label='Density (kg/m³)')
    ax3.set_title(f'Neutral Density at {alt_query} km')
    ax3.set_xlabel('Date (UTC)')
    # set xticks to be 45 degrees
    plt.xticks(rotation=45)
    # ax3.set_ylabel('Latitude (°)')
    ax3.set_ylabel('LST (hours)')
    ax3.grid(True, color='white', linewidth=0.16)



    plt.tight_layout()
    plt.show()

    # plot rho0, H, and beta
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(results['rho0_vec'].T, aspect='auto', origin='lower', extent=[tvec[0], tvec[-1], lat_bins[0], lat_bins[-1]], cmap='jet')#, interpolation = 'gaussian')
    # plt.imshow(results['rho0_vec'].T, aspect='auto', extent=[lat_bins[0], lat_bins[-1], tvec[0], tvec[-1]], origin='lower')
    plt.colorbar(label='Density (kg/m³)')
    plt.title('BC-scaled rho0')
    # plt.xlabel('Date (UTC)')
    plt.ylabel('Latitude (°)')
    plt.subplot(2, 1, 2)
    plt.imshow(results['H_vec'].T, aspect='auto', origin='lower', extent=[tvec[0], tvec[-1], lat_bins[0], lat_bins[-1]], cmap='jet')#, interpolation ='gaussian')
    plt.colorbar(label='Scale Height (km)')
    plt.title('Scale Height')
    # plt.xlabel('Date (UTC)')
    plt.ylabel('Latitude (°)')
    # plt.subplot(3, 1, 3)
    # plt.hist(results['beta_vec'])
    # plt.title('Estimated Beta')
    # plt.xlabel('Beta')
    # plt.ylabel('Frequency')
    # plt.tight_layout()
    plt.show()

    # make a plot of the combined density considering beta, rho0, and H
    beta_avg = 6e-8#1e-7
    alt_query = 500
    dens_2d = (1/beta_avg)* results['rho0_vec'].T  * np.exp(-(alt_query-h0) / results['H_vec'].T) / 1e9 # convert rho0 to kg/m³
    plt.figure(figsize=(12, 6))
    plt.imshow(dens_2d, aspect='auto', origin='lower', extent=[tvec[0], tvec[-1], lat_bins[0], lat_bins[-1]], cmap='jet')#, interpolation = 'gaussian')
    plt.colorbar(label='Density (kg/m³)')
    plt.title(f'Neutral Density at {alt_query} km')
    plt.xlabel('Date (UTC)')
    plt.ylabel('Latitude (°)')
    plt.show()

    # plot a 3x3 grid of density profiles at different altitudes from 400 to 1000 km
    h_query_vec = np.linspace(400, 1000, 9)
    plt.figure(figsize=(12, 8))
    for i, h_query in enumerate(h_query_vec):
        dens_query = (1/beta_avg)* results['rho0_vec'].T  * np.exp(-(h_query-h0) / results['H_vec'].T) / 1e9 # convert rho0 to kg/m³
        plt.subplot(3, 3, i + 1)
        plt.imshow(dens_query, aspect='auto', origin='lower', extent=[tvec[0], tvec[-1], lat_bins[0], lat_bins[-1]], cmap='jet')
        # don't show x axis labels
        plt.xticks([])
        plt.colorbar(label='Density (kg/m³)')
        plt.title(f'Neutral Density at {h_query:.0f} km')
        plt.ylabel('Latitude (°)')
    plt.tight_layout()
    # plt.savefig(f"{output_dir}/density_profiles_all_altitudes.png")
    plt.show()



                                                                 
if __name__ == "__main__":
    main()

