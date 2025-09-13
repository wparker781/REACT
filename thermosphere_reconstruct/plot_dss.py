import numpy as np
from scipy.sparse import diags, vstack, csr_matrix
from scipy.sparse.linalg import lsqr
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import pickle as pkl
import os
from tqdm import tqdm
import time
import matplotlib.colors as mcolors
# set font to arial
plt.rcParams['font.family'] = 'Arial'


# === Main Function ===
def main():
    t1 = time.time()
    t_min = 30 # time resolution in minutes
    # === Load Data ===
    # file_path = 'density_history_june2015_v2_10m_filt.pkl'
    file_path = 'data/gannon_20240509_' + str(t_min) + 'm_ds.pkl'
    with open(file_path, 'rb') as f:
        data = pkl.load(f)

    output_dir = "temp0730_june2015_v2_10m"
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

    plt.figure()
    ax = plt.gca()
    norm = mcolors.LogNorm(vmin=np.min(ds_list), vmax=np.max(ds_list))
    for i in range(len(ds_list)):
        c = plt.cm.viridis(norm(ds_list[i]))
        plt.plot([time_list[i][0], time_list[i][-1]], [i, i], color=c)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    plt.xticks(rotation=45)
    plt.colorbar(sm, ax=ax, label=r'd$_s$ (km$^2$/kg)')
    plt.xlabel('Date (UTC)')
    plt.ylabel('Measurement Index')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()