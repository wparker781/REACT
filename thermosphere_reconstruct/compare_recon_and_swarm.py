# open pickle file and plot the results
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os
import spacepy.pycdf as pycdf
# set font to arial
plt.rcParams['font.family'] = 'Arial'
from scipy.spatial import cKDTree
from scipy.interpolate import griddata


def main():
    version = '1d'

    if version == '1d':
        # open the pickle file
        file_path = 'temp0730_june2015_v2_10m/coord_descent_results.pkl'
        with open(f"{file_path}", 'rb') as f:
            results = pkl.load(f)

        tvec = results['tvec']
        rho0_vec = results['rho0_vec']
        H_vec = results['H_vec']
        dens = results['dens']
        loss_history = results['loss_history']
        beta_vec = results['beta_vec']

        dir = '../ml_decay_prediction/react_thesis/swarm_data_C_pod_june2015'
        swarm_dens, swarm_time, swarm_lat, swarm_lon, swarm_lst, swarm_alt = load_swarm_dens(dir)

        # for each timestep in swarm_time, find the closest time in tvec
        swarm_time_idx = np.zeros(len(swarm_time), dtype=int)
        for i in range(len(swarm_time)):
            swarm_time_idx[i] = np.argmin(np.abs(tvec - swarm_time[i]))
        # get the corresponding rho0 and H for each swarm_time
        rho0_swarm = rho0_vec[swarm_time_idx]
        H_swarm = H_vec[swarm_time_idx]
        # compute the density based on rho0 and H for each swarm_time
        dens_recon = rho0_swarm * np.exp(-(swarm_alt - 600) / H_swarm)

        # find the multiplier on dens_recon that minimizes the squared residuals of the error with dens_swarm
        # minimize the squared residuals between dens_recon and swarm_dens
        i_s = 2500
        i_e = -4000
        a = np.sum(dens_recon[i_s:i_e] * swarm_dens[i_s:i_e]) / np.sum(dens_recon[i_s:i_e] ** 2)

        # plot
        plt.figure()
        plt.plot(swarm_time, swarm_dens, label='Swarm POD', alpha=0.5)
        plt.plot(swarm_time[i_s:i_e], a*dens_recon[i_s:i_e], label='REACT', alpha=0.7)
        plt.xlabel('Date (UTC)')
        plt.ylabel('Density (kg/m³)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    if version == '3d':
        # open the pickle file
        output_dir = "output_lat_lst_time_fit"
        file_path = "dens_results_lat_lst_time_june2015_30m_weighted.pkl"
        # file_path = "dens_results_lat_lst_time.pkl"
        with open(f"{output_dir}/{file_path}", 'rb') as f:
            results = pkl.load(f)

        tvec = results['tvec']
        lat_bins = results['lat_bins']
        lst_bins = results['lst_bins']
        rho0_vec = results['rho0_vec']
        H_vec = results['H_vec']
        loss_history = results['loss_history']
        beta_vec = results['beta_vec']
        counts = results['counts']

    # for i in range(len(tvec)):
    #     plt.figure()
    #     plt.imshow(counts[i,:,:], label='Counts', aspect='auto', extent=[lst_bins[0], lst_bins[-1], lat_bins[0], lat_bins[-1]], origin='lower', cmap='jet')
    #     plt.colorbar(label='Counts')
    #     plt.title(f'Time {tvec[i]}s')
    #     plt.tight_layout()
    #     plt.show()

    # # Step 1: Identify valid and invalid indices
    # valid_mask = counts >= 40
    # invalid_mask = ~valid_mask

    # valid_indices = np.array(np.nonzero(valid_mask)).T
    # invalid_indices = np.array(np.nonzero(invalid_mask)).T

    # # Step 2: Build KDTree from valid cells
    # tree = cKDTree(valid_indices)

    # # Step 3: Find nearest valid cell for each invalid cell
    # _, nearest_idx = tree.query(invalid_indices)

    # # Step 4: Replace rho and H in invalid cells with nearest valid values
    # rho0_vec[invalid_mask] = rho0_vec[tuple(valid_indices[nearest_idx].T)]
    # H_vec[invalid_mask] = H_vec[tuple(valid_indices[nearest_idx].T)]

    # Create coordinate grid
    x, y, z = np.meshgrid(np.arange(len(tvec)), np.arange(len(lat_bins)-1), np.arange(len(lst_bins)-1), indexing='ij')

    # Mask valid cells
    valid_mask = counts >= 20
    invalid_mask = ~valid_mask

    # Flatten coordinates and values for valid points
    points = np.column_stack((x[valid_mask], y[valid_mask], z[valid_mask]))
    rho_values = rho0_vec[valid_mask]
    H_values = H_vec[valid_mask]

    # Interpolate onto full grid (including invalid locations)
    rho_interp = griddata(points, rho_values, (x, y, z), method='nearest')  # or 'linear'
    H_interp = griddata(points, H_values, (x, y, z), method='nearest')

    # Fill invalid cells
    rho0_vec[invalid_mask] = rho_interp[invalid_mask]
    H_vec[invalid_mask] = H_interp[invalid_mask]

    # load swarm density data
    dir = 'swarm_data_C_pod_gannon'
    swarm_dens, swarm_time, swarm_lat, swarm_lon, swarm_lst, swarm_alt = load_swarm_dens(dir)

    # compute density based on rho0_vec and H_vec for each time, lat, lst, and alt from swarm
    rho_recon = np.zeros(len(swarm_time))
    time_idx = np.zeros(len(swarm_time), dtype=int)
    lat_idx = np.zeros(len(swarm_time), dtype=int)
    lst_idx = np.zeros(len(swarm_time), dtype=int)


    for i in range(len(swarm_time)):
        # find the closest time in tvec
        time_idx[i] = np.argmin(np.abs(tvec - swarm_time[i]))-1
        # find the closest lat in lat_bins
        lat_idx[i] = np.digitize(swarm_lat[i], lat_bins) -1
        # find the closest lst in lst_bins
        # lst_idx[i] = np.argmin(np.abs(lst_bins - swarm_lst[i]))-1
        lst_idx[i] = np.digitize(swarm_lst[i], lst_bins) -1
        # get the rho0 and H for this time, lat, lst
        # rho0 = rho0_vec[time_idx[i], lat_idx[i], lst_idx[i]]
        # H = H_vec[time_idx[i], lat_idx[i], lst_idx[i]]        
        rho0 = np.mean(rho0_vec[time_idx[i], lat_idx[i],:])# lst_idx[i]]
        H = np.mean(H_vec[time_idx[i], lat_idx[i],:])# lst_idx[i]]
        # compute the density
        rho_recon[i] = rho0 * np.exp(-(swarm_alt[i] - 600) / H)

    # compute the scale factor to match the swarm density
    i_s = 2000
    i_e = -5000
    a = np.sum(rho_recon[i_s:i_e] * swarm_dens[i_s:i_e]) / np.sum(rho_recon[i_s:i_e] ** 2)


    plt.figure()
    plt.plot(swarm_time, swarm_dens, alpha = 0.7, label='Swarm POD Density')
    plt.plot(swarm_time[i_s:i_e], a*rho_recon[i_s:i_e], alpha = 0.5, label='REACT')
    # print a in the corner
    plt.text(0.05, 0.95, f'Scaling Factor: {a:.2f}', transform=plt.gca().transAxes, fontsize=12)
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

def read_swarm_density(filename):
    with pycdf.CDF(filename) as cdf:
        # Read the density data
        density_data = cdf['density'][:]
        time_data = cdf['time'][:]
        lat_data = cdf['latitude'][:]
        lon_data = cdf['longitude'][:]
        lst_data = cdf['local_solar_time'][:]
        alt_data = cdf['altitude'][:]
    return density_data, time_data, lat_data, lon_data, lst_data, alt_data

def load_swarm_dens(dir):
    # filename = 'swarm_data/SW_OPER_DNSAACC_2__20240501T000000_20240501T235950_0201.cdf'

    # for each file in the swarm_data directory, read the data
    # dir = 'swarm_data_C_pod_gannon'

    # for each file in the directory, read the data
    density = []
    time = []
    lat = []
    lon = []
    lst = []
    alt = []

    for file in os.listdir(dir):
        if file.endswith('.cdf'):
            filepath = os.path.join(dir, file)
            print(f'Reading {filepath}')
            dens_iter, time_iter, lat_iter, lon_iter, lst_iter, alt_iter = read_swarm_density(filepath)
            density.append(dens_iter)
            time.append(time_iter)
            lat.append(lat_iter)
            lon.append(lon_iter)
            lst.append(lst_iter)
            alt.append(alt_iter)
            
            # Do something with the data
            # For example, you can plot or process the data here
            # print(f'Density shape: {density.shape}, Time shape: {time.shape}')

    # Concatenate all data along the first axis
    density = np.concatenate(density, axis=0)
    time = np.concatenate(time, axis=0)
    lat = np.concatenate(lat, axis=0)
    lon = np.concatenate(lon, axis=0)
    lst = np.concatenate(lst, axis=0)
    alt = np.concatenate(alt, axis=0)

    # order time in ascending order, and make the others match
    sort_idx = np.argsort(time)
    density = density[sort_idx]
    time = time[sort_idx]
    lat = lat[sort_idx]
    lon = lon[sort_idx]
    lst = lst[sort_idx]
    alt = alt[sort_idx]/1e3  # convert altitude from m to km


    # clean up large placeholders in density. If value > 1, replace with nan
    density[density > 1] = np.nan
    # alt[alt > 1000] = np.
    
    return density, time, lat, lon, lst, alt


if __name__ == "__main__":
    main()



