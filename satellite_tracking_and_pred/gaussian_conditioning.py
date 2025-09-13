# Re-import after kernel reset
import numpy as np
from traj_predict_tle import get_data_from_tles
import matplotlib.pyplot as plt
# make arial font
plt.rcParams['font.family'] = 'Arial'

def main():

    x_train, y_train, x_test, y_test, satcat, alt_by_obj, dadt, t_train, t_test, dalt_mean, dalt_std, norm_factor_test, f10_mean, f10_std, ap_mean, ap_std = get_data_from_tles()

    # only test on the second half of the year
    # x_test = x_test[int(len(x_test)/2):,:]
    # y_test = y_test[int(len(y_test)/2):,:]
    # t_test = t_test[int(len(t_test)/2):]
    # norm_factor_test = norm_factor_test[int(len(norm_factor_test)/2):,:]

    # Query and support indices
    query_idx = 0
    # support_idx = [1]
    # support_idx_vec = [[], [1], [1,3,6]]
    support_idx_vec = [[], [1], [1,2], [1,2,3], [1,2,3,4], [1,2,3,4,5], [1,2,3,4,5,6], [1,2,3,4,5,6,7], [1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9,10]]
    query_X = x_test

    f10_scaled = x_test[:,0]*f10_std + f10_mean
    ap_scaled = x_test[:,1]*ap_std +  ap_mean

    plt.figure(figsize = (5,4))
    var_theory_lst = []
    var_practice_lst = []
    for i in range(len(support_idx_vec)):
        support_idx = support_idx_vec[i]
        support_Y_test = y_test[:,support_idx]
        support_Y_train = y_train[:,support_idx]
        var_theory, var_practice = get_unc_theory_practice(x_train, y_train, x_test, y_test, query_X, query_idx, support_Y_test, support_Y_train, support_idx)
        # mu_pred, var_pred = predict_satellite_response_v2(
        #     y_train, x_train, query_X, support_Y_test, query_idx, support_idx
        # )
        var_theory_lst.append(var_theory)
        var_practice_lst.append(var_practice)
        # plt.plot(i, var_pred[0],'o')

    plt.plot(range(len(support_idx_vec)), var_theory_lst, 'o-', color = 'firebrick', alpha = 0.6, label = 'Theoretical')
    # plt.plot(range(len(support_idx_vec)), var_practice_lst, 'o-', color = 'steelblue', alpha = 0.6, label = 'Practical')
    plt.xlabel('Number of reference satellites')
    plt.ylabel('Variance of prediction')
    plt.title('Variance of prediction vs number of support satellites')
    plt.grid(axis = 'y', color = 'lightgray',  linewidth = 0.5)
    # plt.ylim([0,1.1])
    plt.tight_layout()
    # plt.legend()
    plt.show()

    plt.figure(1, figsize = (4,8))
    for i in range(len(support_idx_vec)):
        support_idx = support_idx_vec[i]
        support_Y_test = y_test[:,support_idx]
        support_Y_train = y_train[:,support_idx]
        # mu_pred, std_pred = model_with_unc(
        #     x_train, y_train, x_test, y_test,query_X, query_idx, support_Y_test, support_Y_train, support_idx
        # )        
        mu_pred, std_pred = model_with_unc(
            x_train, y_train, x_test,query_X, query_idx, support_Y_test, support_Y_train, support_idx
        )

        # normalize mu_pred and std_pred according to norm factor test and dalt_mean, dalt_std
        # mu_pred = (mu_pred * dalt_std[query_idx] + dalt_mean[query_idx])*norm_factor_test[:,query_idx]
        # std_pred = (std_pred * dalt_std[query_idx] + dalt_mean[query_idx])*norm_factor_test[:,query_idx]
        # y_test_trans = (y_test[:,query_idx] * dalt_std[query_idx] + dalt_mean[query_idx])*norm_factor_test[:,query_idx]

        # for plotting without the normalization, comment above and uncomment below
        y_test_trans = y_test[:,query_idx] #(y_test[:,query_idx] * dalt_std[query_idx] + dalt_mean[query_idx])*norm_factor_test[:,query_idx]


        plt.figure(1)
        plt.subplot(len(support_idx_vec)+1, 1, i + 1)
        plt.grid(axis = 'y', color = 'lightgray',  linewidth = 0.5)
        # plt.fill_between(t_train, -1.5e-5, 0.5e-5, color='gainsboro', alpha=0.5)
        plt.fill_between(t_train, -10, 5, color='gainsboro', alpha=0.5)
        # plt.title(f"n_ref_sats: {len(support_idx)}")
        plt.plot(t_test, y_test_trans, 'k', label='True', linewidth = 1)
        plt.plot(t_test, mu_pred, 'r', label='Predicted', linewidth = 1)
        # plt.ylim([-1.5e-5,0.5e-5])
        plt.ylim([-10,5])
        plt.fill_between(t_test, mu_pred - 2 * std_pred, mu_pred + 2 * std_pred, color='r', alpha=0.2)
        # fill a shaded region that spans t_train
        # plt.ylabel(r'$\dot{a}$ [km/s]')
        plt.ylabel(r'$d_{sn}$')
        plt.gca().set_xticklabels([])


    plt.figure(1)
    plt.subplot(len(support_idx_vec)+1, 1, len(support_idx_vec) + 1)  
    # plot F10 on left x axis and ap on right x axis
    plt.plot(t_test, f10_scaled, 'darkorange', label='F10', alpha = 0.8, linewidth = 1.5)
    plt.ylabel('F10.7 [sfu]', color = 'darkorange')
    plt.gca().tick_params(axis='y', labelcolor='darkorange')
    # plt.gca().set_xticklabels([])
    plt.gca().set_ylim([100, 426])
    plt.xticks(rotation=45)
    plt.fill_between(t_train, 100, 426, color='gainsboro', alpha=0.5)
    plt.xlabel('Date')
    ax2 = plt.gca().twinx()
    ax2.plot(t_test, ap_scaled, 'teal', label='ap', alpha = 0.8, linewidth = 1.5) 
    ax2.tick_params(axis='y', labelcolor='teal')
    ax2.set_ylabel('ap', color = 'teal')
    # ax2.set_ylim([0, 400])
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    # reduce vertical space between subplots
    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.2)
    plt.show()



def get_unc_theory_practice(x_train, y_train, x_test, y_test, query_X, query_idx, support_Y_test, support_Y_train, support_idx):
    # compute theoretical uncertainty on training data
    mean_theory, var_theory = predict_satellite_response_v2(y_train, x_train, x_train, support_Y_test, query_idx, support_idx)
    resid = mean_theory - y_train[:,query_idx]
    # compute variance of residuals
    var_practice = np.var(resid)

    return var_theory[0], var_practice



def model_with_unc(x_train, y_train, x_test, query_X, query_idx, support_Y_test, support_Y_train, support_idx):
    # Predict
    mu_pred, var_pred = predict_satellite_response_v2(
        y_train, x_train, query_X, support_Y_test, query_idx, support_idx
    )

    # assess on training data
    mu_train, var_train = predict_satellite_response_v2(
        y_train, x_train, x_train, support_Y_train, query_idx, support_idx
    )

    resid = mu_train - y_train[:,query_idx]
    sq_resid = resid**2
    # normalize sq_resid
    sq_resid_n = (sq_resid - np.mean(sq_resid)) / np.std(sq_resid)

    # find W that maps from x_test to sq_resid
    W = np.linalg.lstsq(x_train, sq_resid_n, rcond=None)[0]

    var_pred2_n = x_test @ W
    var_pred2 = var_pred2_n * np.std(sq_resid) + np.mean(sq_resid)
    std_pred = np.sqrt(var_pred2)

    # replace any values in std_pred that are nan with zero
    std_pred[np.isnan(std_pred)] = 0

    return mu_pred, std_pred

def predict_satellite_response(Y_train, X_train, query_X, support_Y, query_idx, support_idx):
    n_sats, t_train = Y_train.shape
    n_drivers = X_train.shape[0]

    YX = np.vstack([Y_train, X_train])
    mu = np.mean(YX, axis=1, keepdims=True)
    YX_centered = YX - mu

    C = np.cov(YX_centered)

    query_full_idx = query_idx
    support_full_idx = support_idx
    X_full_idx = list(range(n_sats, n_sats + n_drivers))
    known_idx = support_full_idx + X_full_idx
    known_vals = np.hstack([support_Y, query_X])
    known_mu = mu[known_idx].flatten()

    C_UU = C[query_full_idx, query_full_idx]
    C_Uknown = C[query_full_idx, known_idx].reshape(1, -1)
    C_knownU = C_Uknown.T
    C_known = C[np.ix_(known_idx, known_idx)]

    delta = known_vals - known_mu
    mu_U = mu[query_full_idx, 0]

    mu_post = mu_U + C_Uknown @ np.linalg.inv(C_known) @ delta.T
    var_post = C_UU - C_Uknown @ np.linalg.inv(C_known) @ C_knownU

    return mu_post[0], var_post[0][0]

def predict_satellite_response_v2(
    Y_train, X_train, query_X_series, support_Y_series, query_idx, support_idx
):
    """
    Predicts the time series response of a query satellite using a multivariate Gaussian model
    conditioned on observed support satellites and space weather inputs.

    Parameters
    ----------
    Y_train : ndarray (t_train, n_sats)
        Training satellite responses over time.
    X_train : ndarray (t_train, n_drivers)
        Space weather drivers over the training period.
    query_X_series : ndarray (t_pred, n_drivers)
        Space weather drivers over the prediction period.
    support_Y_series : ndarray (t_pred, n_support)
        Support satellite responses over the prediction period.
    query_idx : int
        Index of the satellite to predict.
    support_idx : list of int
        Indices of support satellites.

    Returns
    -------
    mu_preds : ndarray (t_pred,)
        Predicted mean values for the query satellite.
    var_preds : ndarray (t_pred,)
        Predicted variance values for the query satellite.
    """

    # Transpose training data to (n_features, t)
    Y_train_T = Y_train.T  # (n_sats, t_train)
    X_train_T = X_train.T  # (n_drivers, t_train)

    n_sats, t_train = Y_train_T.shape
    n_drivers = X_train_T.shape[0]
    t_pred = query_X_series.shape[0]

    # Combine satellite responses and space weather drivers
    YX_train = np.vstack([Y_train_T, X_train_T])
    mu_train = np.mean(YX_train, axis=1, keepdims=True)
    YX_centered = YX_train - mu_train
    C = np.cov(YX_centered)

    # Indexing
    query_full_idx = query_idx
    support_full_idx = support_idx
    X_full_idx = list(range(n_sats, n_sats + n_drivers))
    known_idx = support_full_idx + X_full_idx

    C_UU = C[query_full_idx, query_full_idx]
    C_Uknown = C[query_full_idx, known_idx].reshape(1, -1)
    C_knownU = C_Uknown.T
    C_known = C[np.ix_(known_idx, known_idx)]

    mu_SX = mu_train[known_idx].flatten()
    mu_U = mu_train[query_full_idx, 0]

    mu_preds = np.zeros(t_pred)
    var_preds = np.zeros(t_pred)

    for t in range(t_pred):
        support_Y_t = support_Y_series[t]  # shape: (n_support,)
        query_X_t = query_X_series[t]      # shape: (n_drivers,)
        known_vals = np.hstack([support_Y_t, query_X_t])
        delta = known_vals - mu_SX

        mu_post = mu_U + C_Uknown @ np.linalg.inv(C_known) @ delta
        var_post = C_UU - C_Uknown @ np.linalg.inv(C_known) @ C_knownU

        mu_preds[t] = mu_post
        var_preds[t] = var_post

    return mu_preds, var_preds

if __name__ == "__main__":
    main()  
