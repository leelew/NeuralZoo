import argparse
import pickle

import numpy as np
import tensorflow as tf
from MetReg.benchmark.benchmark import ScoreBoard

from _data_generator import _get_task_from_regions
from main import _read_inputs


def _predict_1task(X, 
              y,
              task,
              mdl_name='ml.tree.lightgbm',
              save_path='/hard/lilu/saved_models/',):
    # shape
    N, _, nlat, nlon, _ = y.shape

    # init
    y_pred = np.full((N, nlat, nlon), np.nan)


    if mdl_name.split('.')[0] == 'ml':
        f = open(save_path + mdl_name + '/saved_model_' +
                str(task) + '.pickle', 'rb')
        saved_model = pickle.load(f)

        # predict
        for i in range(nlat):
            for j in range(nlon):
                mdl = saved_model[i][j]
                if mdl is not None:
                    y_pred[:,i,j] = mdl.predict(X[:,:,i,j,:].reshape(N, -1))

    elif mdl_name.split('.')[0] == 'sdl':

        mdl = tf.keras.models.load_model(
            save_path+mdl_name+'/saved_model_'+str(task))
        y_pred = np.squeeze(mdl.predict(X))

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
        y_pred[:, attr[0]:attr[0]+18, attr[1]:attr[1]+18] = y_pred_+mask.reshape(1, 18,18).repeat(N, axis=0)
        y_true[:, attr[0]:attr[0]+18, attr[1]:attr[1]+18] = y_true_+mask.reshape(1, 18,18).repeat(N, axis=0)
        
    np.save(mdl_name.split('.')[-1]+'_pred.npy', y_pred)
    np.save(mdl_name.split('.')[-1]+'_true.npy', y_true)


def benchmark(mdl_name, 
              save_path='/work/lilu/MetReg/experiments/ERA5/'):
    

    y_pred = np.load(mdl_name.split('.')[-1]+'_pred.npy')
    y_true = np.load(mdl_name.split('.')[-1]+'_true.npy')

    sb = ScoreBoard(mode=-1)
    score = sb.benchmark(y_true, y_pred)
    np.save(mdl_name.split('.')[-1]+'_score.npy', score)


if __name__ == "__main__":

    parse = argparse.ArgumentParser()
    parse.add_argument('--mdl_name', type=str, default='ml.lr.ridge')
    config = parse.parse_args()

    _predict(mdl_name=config.mdl_name)
    benchmark(mdl_name=config.mdl_name)

    """
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
