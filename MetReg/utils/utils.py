# -----------------------------------------------------------------------------
#                                Utils module                                 #
# -----------------------------------------------------------------------------
# author: Lu Li                                                               #
# mail: lilu35@mail2.sysu.edu.cn                                              #
# -----------------------------------------------------------------------------
# This Repo is train module of artifical intelligence models, it has three    #
# -----------------------------------------------------------------------------


import functools
import os
import pickle
import time
import glob
#import netCDF4
import numpy as np
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_squared_error, median_absolute_error,
                             r2_score)

import h5py


def get_koppen_index():
    koppen_index = np.array(h5py.File('koppen_index.mat')['koppen_index']).T
    koppen_index = np.concatenate(
        (koppen_index[:, 361:], koppen_index[:, :361]), axis=-1)


def _read_inputs(task,
                 input_path='/WORK/sysu_yjdai_6/hard/lilu/ERA5_1981_2017_DD_A1/',
                 mask_path='/WORK/sysu_yjdai_6//hard/lilu/ERA5_1981_2017_DD_A1/',):
    # load pickle
    f = open(input_path + 'ERA5_DD_A1_case_' + str(task) + '.pickle', 'rb')
    inputs = pickle.load(f)
    # get train/validate set
    X_train = inputs['X_train']
    X_valid = inputs['X_valid']
    y_train = inputs['y_train']
    y_valid = inputs['y_valid']

    # load pickle
    f = open(mask_path + 'nan_mask_case_' + str(task) + '.pickle', 'rb')
    mask = pickle.load(f)

    return X_train, X_valid, y_train, y_valid, mask


def _read_nc(file_path):
    """load and process single netcdf in ERA5 case.

    Args:
        file_path ([type]): [description]
    """
    obj = netCDF4.Dataset(file_path)

    # Notes:
    #   may be raise keyerror, ensure the key of target variables must
    #   lay the last of variables keys sets.
    targ_var_name = list(obj.variables.keys())[-1]
    targ_var = obj[targ_var_name][:]

    # Notes:
    #   range of fill value must be noticed, which is case-different.
    targ_var[targ_var < 1e-4] = np.nan

    # get shape of raw data
    Ntime, Nlat, Nlon = targ_var.shape

    # init average matrix and average spatiotemporally
    avg_spatial_targ_var = np.full((Ntime, Nlat//4, Nlon//4), np.nan)
    for i in range(Nlat // 4):
        for j in range(Nlon // 4):
            _spatial_targ_var = targ_var[:, 4 * i:4 * i + 4, 4 * j:4 * j + 4]
            avg_spatial_targ_var[:, i, j] = np.nanmean(
                _spatial_targ_var, axis=(-1, -2))

    avg_time_spatial_targ_var = np.full((Ntime//24, Nlat//4, Nlon//4), np.nan)
    for t in range(Ntime // 24):
        _time_targ_var = avg_spatial_targ_var[24 * t:24 * t + 24, :, :]
        avg_time_spatial_targ_var[t, :, :] = np.mean(_time_targ_var, axis=0)

    return avg_time_spatial_targ_var


def _get_folder_list(folder_path, file_type='nc'):
    """Get list of files in target folder.

    Args:
        folder_path (str):
            path of main fold of files
        file_type (str):
            type of files in folder
    """
    # get list
    l = glob.glob(folder_path + 'ERA5*' + file_type, recursive=True)
    print(l)

    # sort list
    num_in_str = []

    # TODO: split name automatically, rather than manually.
    for i, file_path in enumerate(l):
        year = file_path.split('_')[2]
        month = file_path.split('_')[3]
        num_in_str.append(int(year + month))

    # index in order
    indices = np.argsort(np.array(num_in_str))

    # order list
    sorted_l = [l[i] for i in indices]
    return sorted_l


def _get_task_from_regions(num_lat, num_lon, interval):
    """Generate begin index of lat & lon"""
    # config
    lat = np.arange(0, num_lat-1, interval)
    lon = np.arange(0, num_lon-1, interval)
    # generate region
    return [[int(lat[i]), int(lon[j])] for i in range(len(lat))
            for j in range(len(lon))]


def _get_nan_mask(y=None):
    """"get mask image for NaN position of inputs."""
    mask = np.full((y.shape[-2], y.shape[-1]), 0.0)

    for i in range(y.shape[-2]):
        for j in range(y.shape[-1]):
            if np.isnan(y[:, i, j]).any():
                mask[i, j] = np.nan
    return mask


def print_log():
    """Basic info"""
    print('welcome to deep learning world \n')
    print('      __  __     _______     _______     ____       _______')
    print('     |  \/  |   |  _____|   |___ ___|   |  _  |    |  _____|')
    print('     | \  / |   | |_____       | |      | |_|_|    | |_____ ')
    print('     | |\/| |   |  _____|      | |      | | \ \    |  _____|')
    print('     | |  | |   | |_____       | |      | |  \ \   | |_____')
    print('     |_|  |_|   |_______|      |_|      |_|   \_\  |_______|')
    print('\n[MetReg][INFO] @author: Lu Li')
    print('[MetReg][INFO] @mail: lilu35@mail2.sysu.edu.cn \n')


def inverse(pred, true):
    """Inverse prediction array with true array"""
    # scaler
    scaler = MinMaxScaler()
    # inverse
    for i in range(true.shape[1]):
        for j in range(true.shape[2]):
            if (np.isnan(true[:, i, j, :]).any()) or (np.isnan(pred[:, i, j, :]).any()):
                print('This pixel ')
            else:
                scaler.fit_transform(true[:, i, j, :])
                pred[:, i, j, :] = scaler.inverse_transform(pred[:, i, j, :])
    return pred


def save2pickle(data, out_path, out_file):
    """Save to pickle file"""
    # if not have output path, mkdir it
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    # save to pickle
    handle = open(out_path + out_file, 'wb')
    pickle.dump(data, handle, protocol=4)
    handle.close()


def save(mdl: dict, dir_save, name_save):
    """save sklearn model.

    Args:
        mdl (dict): [description]
        dir_save ([type]): [description]
        name_save ([type]): [description]
    """
    if not os.path.isdir(os.getcwd() + dir_save):
        os.mkdir(os.getcwd() + dir_save)

    pickle.dump(mdl, open(os.getcwd() + dir_save + name_save, 'wb'))


def tictoc(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        print('Time cost is: %d' % (end_time-start_time))
        return res
    return wrapper


def get_ml_models(block):
    """[summary]

    Args:
        block ([type]): [description]

    Raises:
        NotImplementedError: [description]
    """
    models = []

    for mdl in block:
        if 'linear' in mdl:
            lr = Linear_Regression()
            if 'default' in mdl:
                models.append(lr.default())
            elif 'lasso' in mdl:
                models.append(lr.lasso())
            elif 'ridge' in mdl:
                models.append(lr.ridge())
            elif 'elastic' in mdl:
                models.append(lr.elasticnet())
            else:
                raise NotImplementedError(
                    r"This linear model hasn't been implemented")
        elif 'svr' in mdl:
            svr = Support_Vector_Regression()
            if 'gs' in mdl:
                models.append(svr.gridsearch(
                    svr.svr(), tuned_params(algorithm=1)))
            else:
                models.append(svr.svr())
        elif 'tree' in mdl:
            tree = Tree_Regression()
            if 'gs' in mdl:
                models.append(tree.gridsearch(tree.bagging(
                    algorithm=1), tuned_params(algorithm=2)))
            elif 'dt' in mdl:
                models.append(tree.decision_tree())
            elif 'rf' in mdl:
                models.append(tree.bagging(algorithm=1))
            elif 'adaboost' in mdl:
                models.append(tree.boosting(algorithm=1))
            elif 'GBDT' in mdl:
                models.append(tree.boosting(algorithm=2))
            elif 'Xgboost' in mdl:
                models.append(tree.boosting(algorithm=3))
            elif 'LightGBM' in mdl:
                models.append(tree.boosting(algorithm=4))
            else:
                raise NotImplementedError(
                    r"This tree model hasn't been implemented")
        elif 'gpr' in mdl:
            gpr = Gaussian_Process_Regression()
            models.append((gpr.GPR()))
        elif 'nn' in mdl:
            nn = Nerual_Network_Regression()
            models.append((nn.ANN()))
        else:
            raise NameError('please input a correct model name!')

    return models


# reshape data, change 8915,10,8,8,3 - 8915,30, 8,8
def ML_data(inputs):
    """generate data type for machine learning process.

    Args:
        inputs (numpy array): shape as (sample, timesteps, lat, lon, feature)

    Returns:
        outputs (numpy array): shape as (sample, timesteps * feature, lat, lon)
    """
    if inputs.ndim < 4 or inputs.ndim > 5:
        raise TypeError('must be 4 or 5 dimensions numpy matrix.')
    elif inputs.ndim == 4:
        S, LAT, LON, F = inputs.shape
        return np.squeeze(inputs.reshape(S, F, LAT, LON))
    elif inputs.ndim == 5:
        S, T, LAT, LON, F = inputs.shape
        return np.squeeze(inputs.reshape(S, T*F, LAT, LON))


def tuned_params(algorithm=None):
    """
    Generate tuned parameters range for grid search for MLs
    1. support vector regression
    The important parameters of support vector regression are gamma & C & kernel.
    if gamma is very small, the model is too constrained and can't capture the 
    complexity of data. On the contrary, when gamma is large, it can't be able
    to prevent overfitting.

    The C parameters trades off correct classification of training examples 
    against maximization of the decision func's margin. large C means more 
    accurate model, vice versa.

    2. tree regression
    """
    if algorithm == 1:
        tuned_params = {'C': np.logspace(-5, 5, 2),
                        'gamma': np.logspace(-5, 5, 2), }
        # 'kernel': ['poly', 'rbf', 'sigmoid']}

    if algorithm == 2:
        tuned_params = {'n_estimators': np.arange(50, 150, 50)}

    return tuned_params


def cal_metrics(y_predict, y_valid):
    """This ensemble is give performance of image prediction task, rather than
    time series prediction task. The shape of inputs is (Nlat, Nlon, timestep)
    NOTE: We used all metrics provided from ILAMB, which is a contrast model for 
    land surface model.

    Returns:
        [type]: [description]
    """
    # init all metrics
    evs = np.full((y_predict.shape[-1], y_predict.shape[-1]), np.nan)
    mae = np.full((y_predict.shape[-1], y_predict.shape[-1]), np.nan)
    mdae = np.full((y_predict.shape[-1], y_predict.shape[-1]), np.nan)

    mse = np.full((y_predict.shape[-1], y_predict.shape[-1]), np.nan)
    r2 = np.full((y_predict.shape[-1], y_predict.shape[-1]), np.nan)

    print(evs.shape)

    for i in range(y_predict.shape[-1]):
        for j in range(y_predict.shape[-1]):

            # generate metrics
            evs[i, j] = explained_variance_score(
                y_valid[:, i, j], y_predict[:, i, j])
            mae[i, j] = mean_absolute_error(
                y_valid[:, i, j], y_predict[:, i, j])
            mse[i, j] = mean_squared_error(
                y_valid[:, i, j], y_predict[:, i, j])
            mdae[i, j] = median_absolute_error(
                y_valid[:, i, j], y_predict[:, i, j])
            r2[i, j] = r2_score(y_valid[:, i, j], y_predict[:, i, j])

    metrics = {'evs': evs, 'mae': mae, 'mse': mse,
               'mdae': mdae, 'r2': r2}

    return metrics


def gen_callbacks(self):
    # Tensorboard, earlystopping, Modelcheckpoint
    if not os.path.exists(self.logdir):
        os.mkdir(self.logdir)

    output_model_file = os.path.join(self.logdir, self.output_model)

    self.callbacks = [
        tf.keras.callbacks.TensorBoard(self.logdir),
        tf.keras.callbacks.ModelCheckpoint(
            output_model_file, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)]


def plot_learning_curves(self):

    pd.DataFrame((self.history.history)).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 0.1)
    plt.show()
