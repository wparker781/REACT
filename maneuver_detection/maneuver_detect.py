import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import pickle as pkl
import datetime as dt
import os
import math
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
# set font to arial
plt.rcParams['font.family'] = 'Arial'
from scipy.ndimage import generic_filter


def main():
    y_train, satcat, alt_by_obj, t_train = get_data_from_tles()

    n_std = 6

    # plot the debris objects in gray and a reference object in red
    # compute the mean and std at each timestep of y_test for i < 10
    mean_deb = np.mean(y_train[:,:10], axis = 1)
    std_deb = np.std(y_train[:,:10], axis = 1)

    # find the indices where y_test[:,idx_ref]-mean_deb > or < +/-6*std_deb
    idx_ref = 16
    satcat_ref = satcat[idx_ref]

    std_deb = np.squeeze(std_deb)
    err_sat = y_train[:,idx_ref] - mean_deb 
    err_sat = np.squeeze(err_sat)
    idx_prograde = (np.where(err_sat > n_std*std_deb))[0]
    idx_retrograde = (np.where(err_sat < -n_std*std_deb))[0]
    
    plt.figure(figsize = (7,3.5))
    # plot debris trajectories and the mean and std bounds
    # plt.plot(t_train[:-1], mean_deb, 'k-', label='30-day rolling mean')
    # plt.fill_between(t_train[:-1], mean_deb + n_std*std_deb, mean_deb - n_std*std_deb, color = 'lightgray', alpha = 0.5)
    plt.plot(t_train[:-1], y_train[:,:10], color = 'black', alpha = 0.3)
    plt.xlim([dt.date(2024,4,1), dt.date(2024,6,1)])
    # plt.ylim([-7,7])
    plt.ylabel(r'$d_{sn}$')
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize = (7,3.5))
    if len(idx_prograde) > 0:
        for j in range(len(idx_prograde)):
            plt.axvline(x=t_train[idx_prograde[j]], color='red', linestyle='-', alpha = 0.1, zorder = 1)
    if len(idx_retrograde) > 0:
        for j in range(len(idx_retrograde)):
            plt.axvline(x=t_train[idx_retrograde[j]], color='tab:blue', linestyle='-', alpha = 0.1, zorder = 1)

    # plt.subplot(2,1,1)
    for i in range(len(alt_by_obj[0,:])):
        # if i < 10:
            # if satcat[i] != 34648:
            # plt.plot(t_train[:-1], y_train[:,i].T, color = 'black', alpha = 0.2, linewidth = 1)
        # plt.plot(t_train[:-1], mean_deb, 'k-')
        plt.fill_between(t_train[:-1], mean_deb + n_std*std_deb, mean_deb - n_std*std_deb, color = 'lightgray', alpha = 0.5)
        plt.plot(t_train[:-1], mean_deb + n_std*std_deb, 'gray', linewidth = 0.5)
        plt.plot(t_train[:-1], mean_deb - n_std*std_deb, 'gray', linewidth = 0.5)
        if satcat[i] == satcat_ref:
            print(satcat[i])
            plt.plot(t_train[:-1], y_train[:,i].T, color = 'k', linewidth = 2, zorder = 10)
    plt.ylim([-7,7])
    # set x limits to be 1/1/24 to 6/1/24
    plt.xlim([dt.date(2024,4,1), dt.date(2024,6,1)])
    # plt.grid(axis = 'both', color = 'lightgray',  linewidth = 0.5)
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    # get rid of x labels
    # plt.gca().set_xticklabels([])
    plt.ylabel(r'$d_{sn}$')
    plt.tight_layout()
    plt.show()

    # remove instances of maneuvers which are less than 5 consecutive timesteps
    # idx_prograde = np.array([idx_prograde[i] for i in range(len(idx_prograde)) if i == 0 or idx_prograde[i] - idx_prograde[i-1] > 5])
    # idx_retrograde = np.array([idx_retrograde[i] for i in range(len(idx_retrograde)) if i == 0 or idx_retrograde[i] - idx_retrograde[i-1] > 5])
    idx_prograde = keep_long_runs(idx_prograde, 5)
    idx_retrograde = keep_long_runs(idx_retrograde, 5)
        
    t_train = np.array(t_train)

    plt.figure()

    plt.plot(t_train[:-1], y_train[:,idx_ref]-mean_deb, 'k')
    plt.plot(t_train[:-1], n_std*std_deb, 'k', linewidth = 0.5)
    plt.plot(t_train[:-1], -n_std*std_deb, 'k', linewidth = 0.5)
    if len(idx_prograde) > 0:
        plt.plot(t_train[idx_prograde], y_train[idx_prograde,idx_ref]-mean_deb[idx_prograde], 'r.', markersize = 8)
    if len(idx_retrograde) > 0:
        plt.plot(t_train[idx_retrograde], y_train[idx_retrograde,idx_ref]-mean_deb[idx_retrograde], 'b.', markersize = 8)

    # plt.fill_between(t_test, -6*(std_deb), 6*(std_deb), 'lightgray')
    plt.grid(axis = 'both', color = 'lightgray',  linewidth = 0.5)
    plt.xticks(rotation = 45)
    plt.ylim([-10,10])
    plt.xlim([dt.date(2024,1,1), dt.date(2024,6,1)])
    plt.xlabel('Date')
    plt.ylabel('Error w.r.t. consensus')
    plt.tight_layout()
    # limit spacing between plots
    plt.subplots_adjust(hspace=0.2)
    plt.show()


    plt.figure(figsize = (7,4.5))
    plt.plot(t_train[:-1], alt_by_obj[:-1,idx_ref], color = 'black')
    if len(idx_prograde) > 0:
        plt.plot(t_train[idx_prograde], alt_by_obj[idx_prograde,idx_ref], '|', color = 'red', markersize = 8)
    if len(idx_retrograde) > 0:
        plt.plot(t_train[idx_retrograde], alt_by_obj[idx_retrograde,idx_ref], '|', color = 'blue', markersize = 8)
    plt.xlabel('Date')
    plt.ylabel('Altitude [km]')
    plt.grid(axis = 'both', color = 'lightgray',  linewidth = 0.5)
    plt.xticks(rotation=45)
    # plt.ylim([461.5, 469.5])
    # plt.xlim([dt.date(2024,4,1), dt.date(2024,6,1)])
    plt.tight_layout()
    plt.show()


def keep_long_runs(arr, min_length=5):
    arr = np.sort(np.unique(arr))  # Ensure sorted and unique
    diffs = np.diff(arr)
    
    # Identify break points in consecutive sequences
    breaks = np.where(diffs != 1)[0]
    
    # Start and end indices of runs
    starts = np.insert(breaks + 1, 0, 0)
    ends = np.append(breaks, len(arr) - 1)

    # Collect valid runs
    result = []
    for start, end in zip(starts, ends):
        run = arr[start:end + 1]
        if len(run) >= min_length:
            result.extend(run)

    return np.array(result)


def get_data_from_tles():

    file = 'data/example_objs_interp.pkl' # a bit longer period with starlink and ISS added
    with open(file, 'rb') as f:
        t, alt_by_obj, satcat = pkl.load(f)

    # Convert the dates of interest to numpy datetime64
    tdelta_days =  365
    start_date = dt.datetime(2023, 11, 1)
    end_date = start_date + dt.timedelta(days=tdelta_days)

    # Find the indices of the start and end dates in t
    t_ts = []
    for i in range(len(t)):
        t_ts.append(dt.datetime.timestamp(t[i]))
    t_ts = np.array(t_ts)   
    start_date_ts = dt.datetime.timestamp(start_date)
    end_date_ts = dt.datetime.timestamp(end_date)
    start_idx = np.argmin(np.abs(t_ts - start_date_ts))
    end_idx = np.argmin(np.abs(t_ts - end_date_ts))

    idx_obj = np.arange(0,len(satcat))
    train_days = 365
    alt_by_obj_train = alt_by_obj[start_idx:start_idx + 24*train_days, idx_obj] 
    t_train = t[start_idx:start_idx + 24*train_days-1]
    a = (alt_by_obj_train[:-1,:]+6378.15)
    mu = 398600.4418 # km^3/s^2
    norm_factor_train = np.sqrt(a*mu)
    dalt_train_raw = np.diff(alt_by_obj_train, axis=0)/norm_factor_train#*(398600/(alt_by_obj_train[:-1,:]+6378.15))**(1/2))#/((d_ref.T)*v**(3))
    dalt_mean, dalt_std = np.mean(dalt_train_raw, axis = 0), np.std(dalt_train_raw, axis = 0)
    dalt_train = (dalt_train_raw - dalt_mean) / dalt_std

    return dalt_train, satcat, alt_by_obj, t_train

def _vectorized_fill_with_closest_valid(arr):
    """Fill NaNs in 2D array with the closest non-NaN values along each column."""
    arr = arr.copy()
    isnan = np.isnan(arr)
    idx = np.where(~isnan, np.arange(arr.shape[0])[:, None], np.nan)
    filled_idx = np.where(isnan, np.nan, np.arange(arr.shape[0])[:, None])

    # Forward fill
    fwd = np.maximum.accumulate(np.where(np.isnan(idx), -np.inf, idx))
    fwd_vals = arr[fwd.astype(int), np.arange(arr.shape[1])]

    # Backward fill
    rev = np.flip(np.maximum.accumulate(np.where(np.isnan(np.flip(idx, axis=0)), -np.inf, np.flip(idx, axis=0))), axis=0)
    back_vals = arr[rev.astype(int), np.arange(arr.shape[1])]

    # Distances
    dist_fwd = np.abs(np.arange(arr.shape[0])[:, None] - fwd)
    dist_back = np.abs(np.arange(arr.shape[0])[:, None] - rev)

    # Choose closer
    use_fwd = dist_fwd <= dist_back
    filled = np.where(isnan, np.where(use_fwd, fwd_vals, back_vals), arr)

    return filled

def _vectorized_rolling_stat_fixed(arr, window, func, axis=0, k=3.0, ddof=0):
    """Generic rolling function with outlier rejection using scipy generic_filter."""
    def wrapped_func(x):
        x = x.reshape(-1)
        if np.all(np.isnan(x)):
            return np.nan
        mean = np.nanmean(x)
        std = np.nanstd(x)
        if std == 0 or np.isnan(std):
            return func(x)
        x = x[np.abs(x - mean) <= k * std]
        if len(x) == 0:
            return np.nan
        return func(x)

    size = (window, 1) if axis == 0 else (1, window)
    result = generic_filter(arr, function=wrapped_func, size=size, mode='nearest')
    return _vectorized_fill_with_closest_valid(result) if axis == 0 else _vectorized_fill_with_closest_valid(result.T).T

def rolling_nanmean(arr, window, axis=0, k=3.0):
    return _vectorized_rolling_stat_fixed(arr, window, np.nanmean, axis=axis, k=k)

def rolling_nanstd(arr, window, axis=0, k=3.0, ddof=0):
    return _vectorized_rolling_stat_fixed(arr, window, lambda x: np.nanstd(x, ddof=ddof), axis=axis, k=k, ddof=ddof)


def read_sw_nrlmsise00(swfile):
    '''
    Parse and read the space weather data

    Usage: 
    sw_obs_pre = read_sw_nrlmsise00(swfile)

    Inputs: 
    swfile -> [str] Path of the space weather data
    
    Outputs: 
    sw_obs_pre -> [2d str array] Content of the space weather data

    Examples:
    >>> swfile = 'sw-data/SW-All.csv'
    >>> sw_obs_pre = read_sw(swfile)
    >>> print(sw_obs_pre)
    [['2020' '01' '07' ... '72.4' '68.0' '71.0']
    ['2020' '01' '06' ... '72.4' '68.1' '70.9']
    ...
    ...
    ['1957' '10' '02' ... '253.3' '267.4' '231.7']
    ['1957' '10' '01' ... '269.3' '266.6' '230.9']]
    '''
    sw_df = pd.read_csv(swfile)  
    sw_df.dropna(subset=['C9'],inplace=True)
    # Sort from newest date to past
    sw_df.sort_values(by=['DATE'],ascending=False,inplace=True)
    sw_df.reset_index(drop=True,inplace=True)
    return sw_df

if __name__ == "__main__":
    main()