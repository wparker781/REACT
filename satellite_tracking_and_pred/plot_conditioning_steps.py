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
    # Simulate space weather data (X) and satellite decay rates (Y)
    # X, Y , satcat, alt, dadt, t = get_data_from_tles()
    x_train, y_train, x_test, y_test, satcat, alt_by_obj, dadt, t_train, t_test, dalt_mean, dalt_std, norm_factor_test, mean_f107, std_f107, mean_aph, std_aph, dalt_train_raw, dalt_train_raw_stats  = get_data_from_tles()
    
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
    norm_alt = True
    # load data
    # with open('data/sat_alt_timeseries_01_01_90_12601.pkl', 'rb') as f:
    #     alt_by_obj, B_list, h0_list, t, t_ts = pkl.load(f)
    # satcat = [str(i) for i in range(1, len(alt_by_obj[0])+1)]

    # file = 'data/tles_resampled_2023-11-01_2024-11-01.pkl' # orig dataset of 12 cosmos-iridium debris objects
    # file = 'data/tles_resampled_linear_2022-07-16_2024-11-01.pkl' # a bit longer period with starlink and ISS added
    file = 'data/example_objs_interp.pkl' # a bit longer period with starlink and ISS added
    with open(file, 'rb') as f:
        t, alt_by_obj, satcat = pkl.load(f)


        
    # with open('data/dens_01_01_90_12601.pkl', 'rb') as f:
    #     dens, lat, lon, alt, t, t_ts = pkl.load(f)

    # plot alt_by_obj vs t
    # plt.figure()
    # for i in range(len(alt_by_obj[0])):
    #     plt.plot(t, alt_by_obj[:,i].T, label = satcat[i])
    # plt.xlabel('Time')
    # plt.ylabel('Altitude (km)')
    # plt.legend()
    # plt.show()

    # also open space weather data csv
    start_date = min(t)
    end_date = max(t)
    # Check to see if appropriate sw file already exists
    if os.path.exists('data/sw_data_'+start_date.strftime('%d_%m_%Y')+'_'+ end_date.strftime('%d_%m_%Y')+'.pkl'):
        with open('data/sw_data_'+start_date.strftime('%d_%m_%Y')+'_'+ end_date.strftime('%d_%m_%Y')+'.pkl', 'rb') as f:
            f107A, f107, Ap, aph, t_sw = pkl.load(f)
    else:
        print('loading sw data...')
        sw_data = read_sw_nrlmsise00('data/SW-All.csv')
        f107A, f107, Ap, aph = get_sw_params(t, sw_data, 0, 0)
        with open('data/sw_data_'+start_date.strftime('%d_%m_%Y')+ '_'+ end_date.strftime('%d_%m_%Y')+'.pkl', 'wb') as f:
            pkl.dump([f107A, f107, Ap, aph, t], f)
    print('loaded all data!')

    # Convert the dates of interest to numpy datetime64
    tdelta_days =  365
    # start_date_np = np.datetime64('2023-11-01')#np.datetime64(start_date)
    # end_date_np = np.datetime64(start_date + dt.timedelta(days=tdelta_days))
    start_date = dt.datetime(2023, 11, 1)
    end_date = start_date + dt.timedelta(days=tdelta_days)

    # Convert the array t to numpy datetime64
    # t_np = np.array(t, dtype='datetime64')

    # don't use object 13 (bad interpolation)
    # satcat = np.delete(satcat, 13)
    # alt_by_obj = np.delete(alt_by_obj, 13, axis = 1)

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
    f107_train = f107[start_idx:start_idx + 24*train_days]
    mean_f107, std_f107 = np.mean(f107_train), np.std(f107_train)
    f107_train = (f107_train - mean_f107) / std_f107
    aph_train = aph[start_idx:start_idx + 24*train_days]
    # create a lagged aph_train and add columns. Make it up to 8 indices behind.
    # n = 5
    # aph_train = np.column_stack((aph_train, np.zeros((len(aph_train), n-1))))
    # for i in range(1, n):
    #     aph_train[:,i] = np.roll(aph_train[:,0], i)

    # mean_aph, std_aph = np.mean(aph_train[:,0]), np.std(aph_train[:,0]) # only need first column since they are all the same, just lagged.
    mean_aph, std_aph = np.mean(aph_train), np.std(aph_train) # only need first column since they are all the same, just lagged.
    aph_train = (aph_train - mean_aph) / std_aph
    t_train = t[start_idx:start_idx + 24*train_days-1]
    if norm_alt == True:
        d_ref = np.zeros((len(idx_obj), len(alt_by_obj_train)-1))
        d_ref_all = np.zeros((len(idx_obj), len(alt_by_obj)-1))
        for i in range(len(idx_obj)):
            d_ref[i,:] = dens_expo(alt_by_obj_train[:-1,i])*1e9 # convert to kg/km^3
            d_ref_all[i,:] = dens_expo(alt_by_obj[:-1,i])*1e9
    else:
        d_ref = np.ones((len(idx_obj), len(alt_by_obj_train)-1))
        d_ref_all = np.ones((len(idx_obj), len(alt_by_obj)-1))  

    # def optimize_exp(alt_by_obj_train, d_ref, idx_debris):
    #     # find the exponent that minimizes the average variance of the decay rates
    #     # i.e. dalt_train_raw = np.diff(alt_by_obj_train, axis=0)/(d_ref.T)**x <-- solve for x!
    #     def obj(x):
    #         dalt_train_raw = np.diff(alt_by_obj_train, axis=0)/(d_ref.T)**x
    #         return np.mean(np.std(dalt_train_raw[:,idx_debris], axis = 1))/np.mean(np.mean(dalt_train_raw[:,idx_debris], axis = 1))
    #     res = minimize(obj, 0, bounds = ((-3,3),))
    #     return 2#res.x

    def compute_v_for_alt(alt_by_obj_train):
        # compute the velocity for a circular orbit with semi-major axis alt_by_obj_train + Re
        r_e = 6378.15 # km
        mu = 398600.4418 # km^3/s^2
        alt = alt_by_obj_train + r_e
        v = np.sqrt(mu/alt)
        return v
    

    
    norm_factor_all = ((alt_by_obj[:-1,:]+6378.15)*(d_ref_all.T)*compute_v_for_alt(alt_by_obj)[:-1,:]**2)

    # idx_debris = [2,3,4,5,6,7,8,9,10,11,12,13]
    # x_opt = optimize_exp(alt_by_obj_train, d_ref, idx_debris)
    v = compute_v_for_alt(alt_by_obj_train)[:-1,:]
    # alt_by_obj_train = 
    # norm_factor_train = ((alt_by_obj_train[:-1,:]+6378.15)*(d_ref.T)*v**2)
    a = (alt_by_obj_train[:-1,:]+6378.15)
    mu = 398600.4418 # km^3/s^2
    norm_factor_train = d_ref.T*np.sqrt(a*mu)
    # compute time between each point in t_train
    # t_train = np.array(t_train, dtype='datetime64[s]')
    t_diff_train_sec = 3600#np.diff(np.array(t_train, dtype='datetime64[s]')).astype(int)[0]
    dadt = np.diff(alt_by_obj_train, axis=0)/t_diff_train_sec
    dalt_train_raw = dadt/norm_factor_train#*(398600/(alt_by_obj_train[:-1,:]+6378.15))**(1/2))#/((d_ref.T)*v**(3))
    dalt_mean, dalt_std = np.mean(dalt_train_raw, axis = 0), np.std(dalt_train_raw, axis = 0)
    # convert any positive dalt_train_raw values to nan

    # plt.figure()
    # plt.plot(dalt_train_raw[:,17], 'k', label='RAW')
    dalt_train_raw_stats = np.where(dalt_train_raw > 0, np.nan, dalt_train_raw)
    # dalt_train_raw_stats = dalt_train_raw
    # plt.plot(dalt_train_raw[:,17], 'r', label='nan')
    # plt.show()

    dalt_mean, dalt_std = np.nanmean(dalt_train_raw_stats, axis = 0), np.nanstd(dalt_train_raw_stats, axis = 0)
    



    # compute 30-day (30*24) rolling nanmean and rolling nanstd for satcat 43613 
    satcat_ref = 43613
    idx_ref = np.where(np.array(satcat) == satcat_ref)[0]
    # mov_nanmean = rolling_nanmean(dalt_train_raw_stats[:,idx_ref],30*24+1)
    # mov_nanstd = rolling_nanstd(dalt_train_raw_stats[:,idx_ref], 30*24+1)

    mov_nanmean = np.squeeze(rolling_nanmean(dalt_train_raw_stats[:, idx_ref], 30 * 24 + 1))
    mov_nanstd = np.squeeze(rolling_nanstd(dalt_train_raw_stats[:, idx_ref], 30 * 24 + 1))


    # mov_nanmean = mov_nanmean[:-360]
    # mov_nanstd = mov_nanstd[:-360]
    # t_train_plt = t_train[:-360]
    # plt.figure(figsize = (6,2.5))
    # plt.grid(axis = 'both', color = 'lightgray', linewidth = 0.5)
    # plt.fill_between(t_train_plt[:-1], mov_nanmean - 3*mov_nanstd, mov_nanmean + 3*mov_nanstd, color='tab:blue', alpha=0.2)
    # plt.plot(t_train_plt[:-1], mov_nanmean, 'k', label='30-day rolling mean')
    # # plt.plot(t_train[:-1], mov_nanstd, 'tab:blue', alpha = 0.7, label='30-day rolling std')
    # # set xticks to 45 deg
    # plt.xticks(rotation = 45)
    # plt.plot(dt.datetime(2024,1,2),0, 'w')
    # plt.plot(dt.datetime(2024,12,31),0, 'w')
    # plt.xlabel('Date')
    # plt.ylabel(r'$d_n$')
    # plt.tight_layout()
    # plt.show()

    # normalize dalt_train
    dalt_train = ((dalt_train_raw_stats - dalt_mean) / dalt_std)

    plt.figure(figsize = (8,6))
    plt.subplot(2,2,1)
    plt.plot(t_train, alt_by_obj_train[:,:10], 'teal', alpha = 0.4)
    plt.ylabel('Altitude [km]')
    plt.xlabel('Date')
    plt.xticks(rotation = 45)

    plt.subplot(2,2,2)
    plt.plot(t_train[:-1], dadt[:,:10], 'teal', alpha = 0.4)
    plt.ylabel('da/dt [km/s]')
    plt.xlabel('Date')
    plt.xticks(rotation = 45)


    plt.subplot(2,2,3)
    plt.plot(t_train[:-1], dalt_train_raw[:,:10], 'teal', alpha = 0.4)
    plt.ylabel(r'$d_s$ [km$^2$/kg]')
    plt.xlabel('Date')
    plt.xticks(rotation = 45)


    plt.subplot(2,2,4)
    plt.plot(t_train[:-1], dalt_train[:,:10], color = 'teal', alpha = 0.4)
    plt.ylabel(r'$d_{sn}$')
    plt.xlabel('Date')
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.show()



    # downsample to every every 24*7 hours
    alt_by_obj_weekly = alt_by_obj[start_idx:start_idx + 24*train_days:24*7, idx_obj]
    d_ref_weekly = d_ref[:,::24*7][:, :-1]
    v_weekly = v[::24*7][:-1,:]
    dalt_weekly_raw = np.diff(alt_by_obj_weekly, axis=0)/((alt_by_obj_weekly[:-1,:])*(d_ref_weekly.T)*v_weekly**2)
    dalt_weekly_mean, dalt_weekly_std = np.mean(dalt_weekly_raw, axis = 0), np.std(dalt_weekly_raw, axis = 0)

    # dalt_train = (dalt_weekly_raw - dalt_weekly_mean) / dalt_weekly_std
    # t_train = t[start_idx:start_idx + 24*train_days:24*7][:-1]

    # break dalt_train into weekly segments. For each segment, compute the correlation matrix. 
    # n_weeks = int(len(t_train)/24/7)
    # C_weekly = np.zeros((n_weeks, len(idx_obj), len(idx_obj)))
    # stationkeeping_check = np.zeros((n_weeks, len(idx_obj)))
    # for i in range(n_weeks):
    #     C_weekly[i,:,:] = np.corrcoef(dalt_train[i*24*7:(i+1)*24*7], rowvar=False)
    #     for j in range(len(idx_obj)):
    #         if np.average(C_weekly[i,j,idx_debris]) < 0.3:
    #             stationkeeping_check[i,j] = 1
    #     print(i/n_weeks)

    n_months = int(len(t_train)/24/30)
    C_weekly = np.zeros((n_months, len(idx_obj), len(idx_obj)))
    stationkeeping_check = np.zeros((n_months, len(idx_obj)))
    for i in range(n_months):
        C_weekly[i,:,:] = np.corrcoef(dalt_train[i*24*30:(i+1)*24*30], rowvar=False)
        # plt.figure()
        # plt.imshow(C_weekly[i,:,:], cmap='viridis')
        # plt.colorbar()
        # plt.show()
        
        # compute the average correlation of the debris objects
        # avg_C_deb = np.mean(C_weekly[i,idx_debris,:][:,idx_debris])

        # for j in range(len(idx_obj)):
        #     if np.average(C_weekly[i,j,idx_debris]) < 0.5*avg_C_deb:
        #         stationkeeping_check[i,j] = 1
        # print(i/n_months)

    # r_f107 = np.zeros(len(idx_obj))
    # r_aph = np.zeros(len(idx_obj))
    # for j in range(len(idx_obj)):
    #     r_f107[j] = np.corrcoef(dalt_train[:,j], f107_train[:-1])[0,1]
    #     r_aph[j] = np.corrcoef(dalt_train[:,j], aph_train[:-1])[0,1]

    # r_mult = r_f107*r_aph


    # set up testing data
    test_days = tdelta_days - train_days
    alt_by_obj_test = alt_by_obj[start_idx:start_idx + 24*(train_days+test_days), idx_obj]
    f107_test = f107[start_idx:start_idx + 24*(train_days+test_days)]
    f107_test = (f107_test - mean_f107) / std_f107
    aph_test = aph[start_idx:start_idx + 24*(train_days+test_days)]
    aph_test = (aph_test - mean_aph) / std_aph
    t_test = t[start_idx:start_idx + 24*(train_days+test_days)]
    t_test = t_test[:-1] # not sure why I have to do this and not just slice in the previous line... 
    if norm_alt == True:
        d_ref_test = np.zeros((len(idx_obj), len(alt_by_obj_test)-1))
        for i in range(len(idx_obj)):
            d_ref_test[i,:] = dens_expo(alt_by_obj_test[:-1,i])
    else:
        d_ref_test = np.ones((len(idx_obj), len(alt_by_obj_test)-1))
    v_test = compute_v_for_alt(alt_by_obj_test)[:-1,:]
    norm_factor_test = ((alt_by_obj_test[:-1,:]+6378.15)*(d_ref_test.T)*v_test**2)
    dalt_test_raw = np.diff(alt_by_obj_test, axis=0)/norm_factor_test#(d_ref_test.T)

    # convert any positive dalt_test_raw values to nan
    # dalt_test_raw[dalt_test_raw > 0] = np.nan
    # normalize using the train mean and std stats
    dalt_test = (dalt_test_raw - dalt_mean) / dalt_std
    
    x_train = np.column_stack((f107_train[:-1], aph_train[:-1], ))
    y_train = dalt_train
    t_train = t[start_idx:start_idx + 24*train_days-1]

    x_test = np.column_stack((f107_test[:-1], aph_test[:-1], ))
    y_test = dalt_test
    t_test = t[start_idx:start_idx + 24*(train_days+test_days)-1][:-1]




    return x_train, y_train, x_test, y_test, satcat, alt_by_obj, np.diff(alt_by_obj, axis=0), t_train, t_test, dalt_mean, dalt_std, norm_factor_test, mean_f107, std_f107, mean_aph, std_aph, dalt_train_raw, dalt_train_raw_stats

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

def get_sw_params(t_dt, sw_data, aph_bias, aph_sd):
    aph = np.zeros(len(t_dt))
    f107A = np.zeros(len(t_dt))
    f107 = np.zeros(len(t_dt))
    Ap = np.zeros(len(t_dt))
    for i in range(len(t_dt)):
        # query the model
        f107A[i],f107[i],Ap[i],aph_obs = get_sw(sw_data,t_dt[i].strftime('%Y-%m-%d'),float(t_dt[i].strftime('%H')))
        hour_of_day = t_dt[i].hour
        hour = np.array([0,3,6,9,12,15,18])
        aph[i] = aph_obs[np.argmin(abs(hour_of_day-hour))]

        # define random deviation on ap
        aph_dev = np.random.normal(aph_bias,aph_sd)
        aph[i] = aph[i] + aph_dev

    return f107A, f107, Ap, aph

def get_sw(sw_df,t_ymd,hour):
    '''
    Extract the necessary parameters describing the solar activity and geomagnetic activity from the space weather data.

    Usage: 
    f107A,f107,ap,aph = get_sw(SW_OBS_PRE,t_ymd,hour)

    Inputs: 
    SW_OBS_PRE -> [2d str array] Content of the space weather data
    t_ymd -> [str array or list] ['year','month','day']
    hour -> []
    
    Outputs: 
    f107A -> [float] 81-day average of F10.7 flux
    f107 -> [float] daily F10.7 flux for previous day
    ap -> [int] daily magnetic index 
    aph -> [float array] 3-hour magnetic index 

    Examples:
    >>> f107A,f107,ap,aph = get_sw(SW_OBS_PRE,t_ymd,hour)
    '''
    ymds = sw_df['DATE']
    j_, = np.where(sw_df['DATE'] == t_ymd)
    j = j_[0]
    f107A,f107,ap = sw_df.iloc[j]['F10.7_OBS_CENTER81'],sw_df.iloc[j]['F10.7_ADJ'],sw_df.iloc[j]['AP_AVG']
    aph_tmp_b0 = sw_df.iloc[j]['AP1':'AP8']   
    # i = int(np.floor_divide(hour,3))
    # ap_c = aph_tmp_b0[i]
    # aph_tmp_b1 = sw_df.iloc[j+1]['AP1':'AP8']
    # aph_tmp_b2 = sw_df.iloc[j+2]['AP1':'AP8']
    # aph_tmp_b3 = sw_df.iloc[j+3]['AP1':'AP8']
    # aph_tmp = np.hstack((aph_tmp_b3,aph_tmp_b2,aph_tmp_b1,aph_tmp_b0))[::-1]
    # apc_index = 7-i
    # aph_c369 = aph_tmp[apc_index:apc_index+4]
    # aph_1233 = np.average(aph_tmp[apc_index+4:apc_index+12])
    # aph_3657 = np.average(aph_tmp[apc_index+12:apc_index+20])
    # aph = np.hstack((ap,aph_c369,aph_1233,aph_3657))
    return f107A,f107,ap,aph_tmp_b0

def dens_expo(h):
    params = [
        (0, 25, 0, 1.225, 7.249),
        (25, 30, 25, 3.899e-2, 6.349),
        (30, 40, 30, 1.774e-2, 6.682),
        (40, 50, 40, 3.972e-3, 7.554),
        (50, 60, 50, 1.057e-3, 8.382),
        (60, 70, 60, 3.206e-4, 7.714),
        (70, 80, 70, 8.77e-5, 6.549),
        (80, 90, 80, 1.905e-5, 5.799),
        (90, 100, 90, 3.396e-6, 5.382),
        (100, 110, 100, 5.297e-7, 5.877),
        (110, 120, 110, 9.661e-8, 7.263),
        (120, 130, 120, 2.438e-8, 9.473),
        (130, 140, 130, 8.484e-9, 12.636),
        (140, 150, 140, 3.845e-9, 16.149),
        (150, 180, 150, 2.070e-9, 22.523),
        (180, 200, 180, 5.464e-10, 29.74),
        (200, 250, 200, 2.789e-10, 37.105),
        (250, 300, 250, 7.248e-11, 45.546),
        (300, 350, 300, 2.418e-11, 53.628),
        (350, 400, 350, 9.518e-12, 53.298),
        (400, 450, 400, 3.725e-12, 58.515),
        (450, 500, 450, 1.585e-12, 60.828),
        (500, 600, 500, 6.967e-13, 63.822),
        (600, 700, 600, 1.454e-13, 71.835),
        (700, 800, 700, 3.614e-14, 88.667),
        (800, 900, 800, 1.17e-14, 124.64),
        (900, 1000, 900, 5.245e-15, 181.05),
        (1000, float('inf'), 1000, 3.019e-15, 268)
    ]
    
    dens = np.zeros(len(h))
    
    for i, h_ellp in enumerate(h):
        for (h_min, h_max, h_0, rho_0, H) in params:
            if h_min <= h_ellp < h_max:
                dens[i] = rho_0 * math.exp(-(h_ellp - h_0) / H)
                break
    
    return dens

if __name__ == "__main__":
    main()