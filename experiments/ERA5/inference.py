import pickle

import numpy as np
from MetReg.benchmark.metrics import nse
#from MetReg.api.model_io import model_benchmarker, model_loader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from _data_generator import _get_task_from_regions
from main import _read_inputs


def _predict_1task(X, 
              y,
              task,
              mdl_name='ml.tree.lightgbm',
              save_path='/hard/lilu/saved_models/',):
    # load pickle
    f = open(save_path + mdl_name + '/saved_model_' +
             str(task) + '.pickle', 'rb')
    saved_model = pickle.load(f)

    # shape
    N, _, nlat, nlon, _ = y.shape

    # init
    y_pred = np.full((nlat, nlon), np.nan)

    # predict
    for i in range(nlat):
        for j in range(nlon):
            mdl = saved_model[i][j]
            if mdl is not None:
                y_pred[:,i,j] = mdl.predict(X[:,:,i,j,:].reshape(N, -1))

    y_true = y.reshape(N, nlat, nlon)
    
    return y_pred, y_true


def _predict(mdl_name):
    # get region and task
    region = _get_task_from_regions(180, 360, 18)

    _,_,_, y_valid, _ = _read_inputs(task=0)

    # shape
    N, _, nlat, nlon, _  = y_valid.shape

    # init
    y_pred = np.full((N, 180, 360), np.nan)
    y_true = np.full((N, 180, 360), np.nan)


    for num_jobs, attr in enumerate(region):

        print('now processing jobs {}'.format(num_jobs))
        X_train, X_valid, y_train, y_valid, mask = _read_inputs(task=num_jobs)
        y_pred_, y_true_ = _predict_1task(X_valid, y_valid, task=num_jobs, mdl_name=mdl_name)
        y_pred[:, attr[0]:attr[0]+18, attr[1]:attr[1]+18] = y_pred_
        y_true[:, attr[0]:attr[0]+18, attr[1]:attr[1]+18] = y_true_

    np.save(mdl_name.split('.')[-1]+'_pred.npy', y_pred)
    np.save(mdl_name.split('.')[-1]+'_true.npy', y_true)






def inference(X,
              y,
              task,
              mdl_name='ml.tree.lightgbm',
              save_path='/hard/lilu/saved_models/',):
    # load pickle
    f = open(save_path + mdl_name + '/saved_model_' +
             str(task) + '.pickle', 'rb')
    saved_model = pickle.load(f)

    # get shape
    N, _, nlat, nlon, _ = y.shape

    spatial_mean = np.full((nlat, nlon), np.nan)
    time_mean = np.full((nlat, nlon), np.nan)
    rmse = np.full((nlat, nlon), np.nan)
    r2 = np.full((nlat, nlon), np.nan)
    mse = np.full((nlat, nlon), np.nan)
    mae = np.full((nlat, nlon), np.nan)
    nse_ = np.full((nlat, nlon), np.nan)

    for i in range(nlat):
        for j in range(nlon):
            mdl = saved_model[i][j]
            if mdl is not None:
                y_predict = mdl.predict(X[:, :, i, j, :].reshape(N, -1))
                r2[i, j] = r2_score(y[:, 0, i, j, 0], y_predict)
                rmse[i,j] = np.sqrt(mean_squared_error(y[:, 0, i, j, 0], y_predict))
                mae[i,j] = mean_absolute_error(y[:, 0, i, j, 0], y_predict)
                nse_[i,j] = nse(y[:, 0, i, j, 0], y_predict)
                spatial_mean[i,j] = np.mean(y[:, 0, i, j, 0])

            else:
                r2[i, j] = np.nan
                rmse[i,j] = np.nan
                mae[i,j] = np.nan


    

    return r2, rmse, mae, nse_, spatial_mean


if __name__ == "__main__":


    _predict(mdl_name='ml.elm.elm')

    """

    region = _get_task_from_regions(180, 360, 18)
    r2_world = np.full((180, 360), np.nan)
    rmse_world = np.full((180, 360), np.nan)
    mae_world = np.full((180, 360), np.nan)
    nse_world = np.full((180, 360), np.nan)
    mean_world = np.full((180, 360), np.nan)

    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument('--mdl_name', type=str, default='ml.lr.ridge')
    config = parse.parse_args()



    for num_jobs, attr in enumerate(region):

        print('now processing jobs {}'.format(num_jobs))
        X_train, X_valid, y_train, y_valid, mask = _read_inputs(task=num_jobs)
        r2, rmse, mae, nse_, spatial_mean = inference(X_valid, y_valid, task=num_jobs, mdl_name=config.mdl_name)
        r2_world[attr[0]:attr[0]+18, attr[1]:attr[1]+18] = r2
        rmse_world[attr[0]:attr[0]+18, attr[1]:attr[1]+18] = rmse
        mae_world[attr[0]:attr[0]+18, attr[1]:attr[1]+18] = mae
        nse_world[attr[0]:attr[0]+18, attr[1]:attr[1]+18] = nse_
        mean_world[attr[0]:attr[0]+18, attr[1]:attr[1]+18] = spatial_mean


    r2_world = np.concatenate((r2_world[:, 181:], r2_world[:, :181]), axis=-1)
    rmse_world = np.concatenate((rmse_world[:, 181:], rmse_world[:, :181]), axis=-1)
    mae_world = np.concatenate((mae_world[:, 181:], mae_world[:, :181]), axis=-1)
    nse_world = np.concatenate((nse_world[:, 181:], nse_world[:, :181]), axis=-1)
    mean_world = np.concatenate((mean_world[:, 181:], mean_world[:, :181]), axis=-1)

    metrics = {
        'r2': r2_world,
        'rmse': rmse_world,
        'mae':mae_world,
        'nse':nse_world,
        'mean': mean_world
        }

    np.save('metric_'+config.mdl_name+'.npy', metrics)

    """