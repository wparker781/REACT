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

def main():
    # Simulate space weather data (X) and satellite decay rates (Y)
    # X, Y , satcat, alt, dadt, t = get_data_from_tles()
    x_train, y_train, x_test, y_test, satcat, alt_by_obj, dadt, t_train, t_test, dalt_mean, dalt_std, norm_factor_test = get_data_from_tles()
    n_timesteps = x_train.shape[0]
    n_satellites = y_train.shape[1]
    n_features = x_train.shape[1]

    # plot t_train vs alt_by_obj
    plt.figure(figsize = (5,4))
    for i in range(len(alt_by_obj[0,:])):
        if i < 10:
            plt.plot(t_test, alt_by_obj[:-1,i].T, label = satcat[i], color = 'tab:orange', alpha = 0.7)
        else:
            plt.plot(t_test, alt_by_obj[:-1,i].T, label = satcat[i], color = 'tab:blue', alpha = 0.7)

    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.ylabel('Altitude [km]')
    plt.title('Altitude of satellites')
    plt.ylim([360,950])
    plt.tight_layout()
    plt.show()


    # combine x_test and y_test into one matrix
    test_ip_op = np.column_stack((x_test, y_test))

    labels_x = ['F10.7', 'ap']
    # make labels y string satcat
    labels_y = [str(i) for i in satcat]

    labels = labels_x + labels_y





    C = np.corrcoef(test_ip_op.T)  # Satellite-to-satellite and space weather correlation

    plt.figure()
    plt.imshow(C, cmap='coolwarm')
    # add numbers in the cells
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            plt.text(j, i, round(C[i, j], 2), ha='center', va='center', color='black', fontsize=6)

    # specify labels on x and y axis as labels
    plt.xticks(np.arange(len(labels)), labels, rotation=90, fontsize=10)
    plt.yticks(np.arange(len(labels)), labels, fontsize=10)

    # Add gridlines to separate cells
    plt.gca().set_xticks(np.arange(-0.5, C.shape[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, C.shape[0], 1), minor=True)
    plt.grid(which="minor", color="black", linestyle='-', linewidth=0.25)
    plt.tick_params(which="minor", bottom=False, left=False)  # Hide minor ticks

    # make colormap coolwarm with range [-1,1]
    plt.clim(-1, 1)


    plt.colorbar(label = 'Correlation coefficient')
    # plt.title('C Space weather and decay rates')
    plt.show()



    # compute correlation matrix for Y
    C_train = np.corrcoef(y_train.T)  # Satellite-to-satellite correlation (5x5)

    # compute W by OLS
    W_ols = np.linalg.pinv(x_train.T @ x_train) @ x_train.T @ y_train  # Shape: (2x5)
    # W_ols_partial = np.linalg.pinv(x_train[:,:2].T @ x_train[:,:2]) @ x_train[:,:2].T @ y_train  # Shape: (2x5)
    y_pred_train = x_train @ W_ols
    y_pred_test = x_test @ W_ols
    resid = np.abs(y_pred_train-y_train)
    resid_sq = (resid)**2
    resid_mean, resid_std = np.mean(resid, axis = 0), np.std(resid, axis = 0)
    resid_norm = (resid - resid_mean) / resid_std
    W_resid = np.linalg.pinv(x_train.T @ x_train) @ x_train.T @ resid_norm  # Shape: (2x5)
    y_pred_resid = x_test @ W_resid
    y_pred_resid_orig = y_pred_resid * resid_std + resid_mean
    # y_pred_resid_orig = np.sqrt(y_pred_resid_orig)

    # convert y_test, y_pred_test, and y_pred_resid_orig to km/hr (original scale) using dalt_mean, dalt_std, and d_ref_test
    y_test = (y_test * dalt_std + dalt_mean)*norm_factor_test
    y_pred_test = (y_pred_test * dalt_std + dalt_mean)*norm_factor_test
    y_pred_resid_orig = (y_pred_resid_orig * dalt_std + dalt_mean)*norm_factor_test


    # plt.figure()
    # plt.plot(t_test, y_pred_resid_orig[:,0], 'r', label='Predicted')
    # plt.plot(t_train, resid[:,0], 'k', label='True')
    # plt.show()

    # compute W for residuals
    # y_pred_partial = x_test[:,:2] @ W_ols_partial

    for i in range(len(y_pred_test[0,:])):
        plt.figure()
        plt.plot(t_test, y_test[:,i], 'k', label='True')
        plt.plot(t_test, y_pred_test[:,i], 'r', label = 'Predicted')
        # fill between y_pred + 2*resid[:,i] and y_pred - 2*resid[:,i]
        plt.fill_between(t_test, y_pred_test[:,i] - 2*y_pred_resid_orig[:,i], y_pred_test[:,i] + 2*y_pred_resid_orig[:,i], color='r', alpha=0.3)
        plt.title('Satellite ' + str(satcat[i]))
        plt.legend()
        plt.show()


    # plt.figure()
    # plt.plot(Y)
    # plt.show()

    # plot only the Y for satcats within satcat_deb
    satcat_deb = [34106, 33874, 34486, 34696, 34648, 34428, 33784, 34473, 33911, 33764]
    incl = [86.4,86.4,86.4,86.4,86.4, 74, 74, 74, 74, 74, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    
    plt.figure(figsize = (5,3))
    plt.subplot(1,3,1)
    for i in range(len(y_test[0,:])):
        if satcat[i] in satcat_deb:
            # get index of satcat in satcat_deb
            satcat_deb_idx = satcat_deb.index(satcat[i])
            if satcat_deb_idx < 5:
                plt.plot(t, alt[:,i], color = 'tab:blue', alpha = 0.7, label = str(satcat[i]))
            else:
                plt.plot(t, alt[:,i], color = 'tab:orange', alpha = 0.7, label = str(satcat[i]))
            # plt.plot(Y[:,i])
            # plt.plot(Y_pred_orig[:,i], 'r', label='Pred ols')
            # plt.title('Satellite ' + str(satcat[i]))
    # plt.legend()
    plt.ylabel('Altitude [km]')
    plt.xlabel('Date')
    plt.xticks(rotation=45)

    # plt.show()

    plt.subplot(1,3,2)
    for i in range(len(Y[0,:])):
        if satcat[i] in satcat_deb:
            # get index of satcat in satcat_deb
            satcat_deb_idx = satcat_deb.index(satcat[i])
            if satcat_deb_idx < 5:
                plt.plot(t[:-1], dadt[:,i], color = 'tab:blue', alpha = 0.5, label = str(satcat[i]))
            else:
                plt.plot(t[:-1], dadt[:,i], color = 'tab:orange', alpha = 0.5, label = str(satcat[i]))
            # plt.plot(Y[:,i])
            # plt.plot(Y_pred_orig[:,i], 'r', label='Pred ols')
            # plt.title('Satellite ' + str(satcat[i]))
    plt.ylabel(r'$da/dt$ [km/hr]')
    plt.xlabel('Date')
    plt.xticks(rotation=45)

    # plt.legend()
    # plt.show()

    # plt.figure()
    plt.subplot(1,3,3)
    for i in range(len(Y[0,:])):
        if satcat[i] in satcat_deb:
            # get index of satcat in satcat_deb
            satcat_deb_idx = satcat_deb.index(satcat[i])
            if satcat_deb_idx < 5:
                plt.plot(t[:-1], Y[:,i], color = 'tab:blue', alpha = 0.5, label = str(satcat[i]))
            else:
                plt.plot(t[:-1], Y[:,i], color = 'tab:orange', alpha = 0.5, label = str(satcat[i]))
            # plt.plot(Y[:,i])
            # plt.plot(Y_pred_orig[:,i], 'r', label='Pred ols')
            # plt.title('Satellite ' + str(satcat[i]))
    # plt.legend()
    plt.ylabel(r'$d_n$')
    plt.xlabel('Date')
    plt.xticks(rotation=45)

    plt.tight_layout()
    # decrease teh horizontal spacing between subplots
    plt.subplots_adjust(wspace=0.5)
    plt.show()

    for i in range(len(Y[0,:])):
        plt.figure()
        plt.plot(Y[:,i],  'k' ,label='True')
        plt.plot(Y_pred_partial[:,i], 'g', label='partial ols')
        plt.plot(Y_pred_orig[:,i], 'r', label='Pred ols')
        plt.title('Satellite ' + str(satcat[i]))

        # plt.fill_between(np.arange(n_timesteps), Y_pred_orig[:,i] - 2*resid[:,i], Y_pred_orig[:,i] + 2*resid[:,i], color='r', alpha=0.2)
        plt.legend()

        plt.show()


    C_y_pred = np.corrcoef(Y_pred_orig.T)  # Satellite-to-satellite correlation of predictions (5x5)

    # plt.figure()
    # plt.imshow(C_y_pred, cmap='viridis')
    # # add numbers in the cells
    # for i in range(C_y_pred.shape[0]):
    #     for j in range(C_y_pred.shape[1]):
    #         plt.text(j, i, round(C_y_pred[i, j], 2), ha='center', va='center', color='black')
    # plt.colorbar()
    # plt.title('C Residuals')
    # plt.show()

    # plt.figure()
    # plt.scatter(Y_pred_orig[:,0], Y[:,0])
    # # also plot a line for X = y
    # plt.plot([0, 1], [0, 1], color='red')
    # plt.xlabel('Y_pred')
    # plt.ylabel('Y_true')
    # plt.show()

    # repeat plot above but make it 2d histogram
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # hist, xedges, yedges = np.histogram2d(Y_pred_orig[:,0], Y[:,0], bins=20, range=[[0, 1], [0, 1]])
    # x_mid = (xedges[1:] + xedges[:-1]) / 2
    # y_mid = (yedges[1:] + yedges[:-1]) / 2
    # x, y = np.meshgrid(x_mid, y_mid)
    # # plot imshow
    # plt.imshow(hist, cmap='viridis')
    # # ax.plot_surface(x, y, hist, cmap='viridis')
    # plt.show()

    resid = Y - Y_pred_orig

    cov_resid = np.cov(resid.T)

    W_upd = W_ols_orig@np.linalg.inv(cov_resid)

    Y_upd = X @ W_upd

    Y_resid_upd = Y - Y_upd

    C_resid_upd = np.corrcoef(Y_resid_upd.T)  # Satellite-to-satellite correlation of predictions (5x5)

    # plt.figure()
    # plt.imshow(C_resid_upd, cmap='viridis')
    # # add numbers in the cells
    # for i in range(C_resid_upd.shape[0]):
    #     for j in range(C_resid_upd.shape[1]):
    #         plt.text(j, i, round(C_resid_upd[i, j], 2), ha='center', va='center', color='black')
    # plt.colorbar()
    # plt.title('C Resid upd')
    # plt.show()


    W_resid = np.linalg.pinv(X.T @ X) @ X.T @ resid  # Shape: (2x5)

    Y_resid = X @ W_resid



    # X = X.T
    # Y = Y.T
    Cov = np.cov(Y.T)
    # compute generalized least squares solution for W where Y = XW when C is the covariance matrix of Y
    cov_inv = np.linalg.inv(Cov)
    Y_gls = Y @ cov_inv
    # W_gls = np.linalg.inv(X.T @ C_inv @ X) @ X.T @ C_inv @ Y  # Shape: (2x5)
    # W_gls_new = np.linalg.inv(X.T@X)@X.T@Y_gls
    W_gls_new = np.linalg.inv(X.T@X)@X.T@Y@np.linalg.inv(cov_resid)

    Y_pred_gls = X @ W_gls_new
    resid_gls = Y - Y_pred_gls
    C_resid_gls = np.corrcoef(resid_gls.T)  # Satellite-to-satellite correlation of predictions (5x5)



    plt.figure()
    plt.imshow(C_resid_gls, cmap='viridis')
    # add numbers in the cells
    for i in range(C_resid_gls.shape[0]):
        for j in range(C_resid_gls.shape[1]):
            plt.text(j, i, round(C_resid_gls[i, j], 2), ha='center', va='center', color='black')
    plt.colorbar()
    plt.title('C Resid GLS')
    plt.show()

    for i in range(n_satellites):
        fig = plt.figure()
        plt.plot(Y[:,i],  'k' ,label='True')
        plt.plot(Y_pred_orig[:,i], 'r', label='Pred ols')
        plt.plot(Y_upd[:,i], 'g', label='Pred upd')
        # plt.fill_between(np.arange(n_timesteps), Y_pred_orig[:,i] - 2*resid[:,i], Y_pred_orig[:,i] + 2*resid[:,i], color='r', alpha=0.2)
        plt.legend()
        plt.show()


    c = 3


def get_data_from_tles():
    mu = 398600.4418  # km^3/s^2
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
    train_days = 120
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
            d_ref[i,:] = dens_expo(alt_by_obj_train[:-1,i])
            d_ref_all[i,:] = dens_expo(alt_by_obj[:-1,i])
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
    norm_factor_train = d_ref.T*1e9*np.sqrt(mu*(alt_by_obj_train[:-1,:]+6378.15))
    dalt_train_raw = (np.diff(alt_by_obj_train, axis=0)/3600)/norm_factor_train#!!! This ASSUMES STATE UPDATES ARE ALAWYS 1 HOUR APART *(398600/(alt_by_obj_train[:-1,:]+6378.15))**(1/2))#/((d_ref.T)*v**(3))
    dalt_mean, dalt_std = np.mean(dalt_train_raw, axis = 0), np.std(dalt_train_raw, axis = 0)
    # normalize dalt_train
    dalt_train = ((dalt_train_raw - dalt_mean) / dalt_std)

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
    # norm_factor_test = ((alt_by_obj_test[:-1,:]+6378.15)*(d_ref_test.T)*v_test**2)
    norm_factor_test = d_ref_test.T*1e9*np.sqrt(mu*(alt_by_obj_test[:-1,:]+6378.15))
    dalt_test_raw = (np.diff(alt_by_obj_test, axis=0)/3600)/norm_factor_test#THIS ASSUMES 1 HOUR UPDATES
    # normalize using the train mean and std stats
    dalt_test = (dalt_test_raw - dalt_mean) / dalt_std
    
    x_train = np.column_stack((f107_train[:-1], aph_train[:-1], ))
    y_train = dalt_train
    t_train = t[start_idx:start_idx + 24*train_days-1]

    x_test = np.column_stack((f107_test[:-1], aph_test[:-1], ))
    y_test = dalt_test
    t_test = t[start_idx:start_idx + 24*(train_days+test_days)-1][:-1]




    return x_train, y_train, x_test, y_test, satcat, alt_by_obj, np.diff(alt_by_obj, axis=0), t_train, t_test, dalt_mean, dalt_std, norm_factor_test, mean_f107, std_f107, mean_aph, std_aph


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