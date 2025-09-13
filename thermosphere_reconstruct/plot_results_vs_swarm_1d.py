# load results and plot
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os
import spacepy.pycdf as pycdf
import sys
sys.path.append('../ml_decay_prediction/react_thesis')
from traj_predict_tle import read_sw_nrlmsise00, get_sw_params 

def main():
    # open pkl file with results
    # === Load results from pkl file
    t_min = 5
    file_path = 'data/gannon_20240509_' + str(t_min) + 'm_ds.pkl'
    # file_path = 'data/octG4_20241006_' + str(t_min) + 'm_ds.pkl'
    output_dir = file_path[:-4] + "_1D_owd"
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


    # load swarm density data
    dir = '../ml_decay_prediction/react_thesis/swarm_data_C_pod_gannon'
    swarm_dens, swarm_time, swarm_lat, swarm_lon, swarm_lst, swarm_alt = load_swarm_dens(dir)


    start_date = min(tvec)#dt.datetime(2024, 5, 9, 0, 0)
    end_date = max(tvec)#dt.datetime(2024, 5, 15, 0, 0)
    # tvec = np.arange(start_date, end_date, dt.timedelta(hours=1))
    # Check to see if appropriate sw file already exists
    if os.path.exists('data/sw_data_1d_'+start_date.strftime('%d_%m_%Y')+'_'+ end_date.strftime('%d_%m_%Y')+'.pkl'):
        with open('data/sw_data_1d_'+start_date.strftime('%d_%m_%Y')+'_'+ end_date.strftime('%d_%m_%Y')+'.pkl', 'rb') as f:
            f107A, f107, Ap, aph, t_sw = pkl.load(f)
    else:
        print('loading sw data...')
        sw_data = read_sw_nrlmsise00('data/SW-All.csv')
        f107A, f107, Ap, aph = get_sw_params(tvec, sw_data, 0, 0)
        with open('data/sw_data_1d_'+start_date.strftime('%d_%m_%Y')+ '_'+ end_date.strftime('%d_%m_%Y')+'.pkl', 'wb') as f:
            pkl.dump([f107A, f107, Ap, aph, tvec], f)
    print('loaded all data!')

    valid = np.where(counts > np.average(counts) * 0.4)[0]

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
    beta_avg = 6e-8 # 8e-8#1.8e-7
    dens_vec = (1 / beta_avg) * rho0_vec * np.exp(-(h_query_vec[:, None] - h0) / H_vec[None, :]) / 1e9

    plt.figure(figsize=(3, 4))
    plt.grid(True, color='lightgray', linewidth=0.25)
    for i, h_val in enumerate(h_query_vec):
        plt.semilogy(tvec[valid], dens_vec[i, valid], label=f"h = {h_val:.0f} km")
    plt.legend()
    plt.xticks(rotation=45)
    plt.xlabel('Date (UTC)')
    plt.ylabel('Neutral Density [kg/m³]')
    # plot twin x with H
    plt.twinx()
    plt.plot(tvec[valid], H_vec[valid], color='k', linestyle = ':', alpha = 0.3, zorder = -1)
    plt.ylabel("H [km]", color = 'darkgray')
    plt.tick_params(axis='y', labelcolor='darkgray')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/density_profiles_all_altitudes.png")

    h_query = 470
    dens = (1 / beta_avg) * rho0_vec * np.exp(-(h_query - h0) / H_vec) / 1e9
    # dens_smooth = np.convolve(dens, np.ones(10) / 10, mode='same')

    plt.figure(figsize=(7, 4))
    plt.grid(True, color='gray', linewidth=0.25)
    plt.plot(swarm_time, swarm_dens, '-', color='k', alpha=0.3, label='Swarm POD Density')
    plt.plot(tvec[valid], dens[valid], c = 'tab:blue', linewidth = 2, label = 'REACT Reconstruction')
    plt.xlabel("Date (UTC)")
    plt.ylabel("Neutral Density [kg/m³]")
    plt.xticks(rotation=45)
    plt.title(f"Density at {h_query} km")
    plt.legend()
    # make ylabel and yticks gray
    plt.twinx()
    plt.plot(tvec, aph, color='red', alpha=0.4, label='ap')
    plt.ylabel('ap', color='red')
    plt.tick_params(axis='y', labelcolor='red')
    plt.gca().tick_params(axis='y', colors='red')
    plt.tight_layout()
    # plt.savefig(f"{output_dir}/density_500km_coord_descent.png")
    plt.show()



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


if __name__ == "__main__":
    main()
