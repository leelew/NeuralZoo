import sys
sys.path.append('../../')

import pickle
import argparse
import numpy as np
import tensorflow as tf
from MetReg.benchmark.benchmark import ScoreBoard
from MetReg.utils.utils import _read_inputs, _get_task_from_regions


"""
def _predict_1task(X,
                   y,
                   task,
                   mdl_name,
                   save_path,):
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
                    y_pred[:, i, j] = mdl.predict(
                        X[:, :, i, j, :].reshape(N, -1))

    elif mdl_name.split('.')[0] == 'sdl':

        mdl = tf.keras.models.load_model(
            save_path+mdl_name+'/saved_model_'+str(task))
        y_pred = np.squeeze(mdl.predict(X))

    else:

        f = open(save_path + mdl_name + '/saved_model_' +
                 str(task) + '.pickle', 'rb')
        log = pickle.load(f)
        y_pred = log['y_pred']
        y_true = log['y_valid']

        return y_pred, y_true

    y_true = y.reshape(N, nlat, nlon)

    return y_pred, y_true


"""


def process(mdl_name, forecast_path, task, threshold=0):

    f = open(forecast_path + mdl_name + '/saved_model_' +
             str(task) + '.pickle', 'rb')
    log = pickle.load(f)
    y_pred = log['y_pred']
    y_true = log['y_valid']

    # get shape
    N, H, W = y_true.shape

    from sklearn.metrics import r2_score

    for i in range(H):
        for j in range(W):
            if not np.isnan(y_true[:, i, j]).any():
                r2 = r2_score(y_true[:, i, j], y_pred[:, i, j])
                if r2 < threshold:
                    y_pred[:, i, j] = np.nanmean(y_pred, axis=(-1, -2))

    return y_pred, y_true


def predict(mdl_name, input_path, forecast_path):
    # get region and task
    region = _get_task_from_regions(180, 360, 18)

    _, _, _, y_valid, _ = _read_inputs(
        task=0, input_path=input_path, mask_path=input_path)

    # shape
    N, _, nlat, nlon, _ = y_valid.shape

    # init
    y_pred = np.full((N, 180, 360), np.nan)
    y_true = np.full((N, 180, 360), np.nan)

    for num_jobs, attr in enumerate(region):

        print('now processing jobs {}'.format(num_jobs))
        X_train, X_valid, y_train, y_valid, mask = _read_inputs(
            task=num_jobs, input_path=input_path, mask_path=input_path)

        y_pred_, y_true_ = process(
            mdl_name=mdl_name, forecast_path=forecast_path,
            task=num_jobs, threshold=0)
        y_pred[:, attr[0]:attr[0]+18, attr[1]:attr[1]+18] = y_pred_ + \
            mask.reshape(1, 18, 18).repeat(N, axis=0)
        y_true[:, attr[0]:attr[0]+18, attr[1]:attr[1]+18] = y_true_ + \
            mask.reshape(1, 18, 18).repeat(N, axis=0)

    np.save(mdl_name.split('.')[-1]+'_pred.npy', y_pred)
    np.save(mdl_name.split('.')[-1]+'_true.npy', y_true)


def benchmark(mdl_name,
              forecast_path,
              score_path):

    y_pred = np.load(forecast_path+mdl_name.split('.')[-1]+'_pred.npy')
    y_true = np.load(forecast_path+mdl_name.split('.')[-1]+'_true.npy')

    sb = ScoreBoard(mode=-1)
    score = sb.benchmark(y_true, y_pred)
    np.save(score_path+mdl_name.split('.')[-1]+'_score.npy', score)

    #sb = ScoreBoard(mode=-1, overall_score=True)
    #score = sb.benchmark(y_true, y_pred)
    # print(score)


if __name__ == "__main__":

    from parser import get_parse
    config = get_parse()

    predict(mdl_name=config.mdl_name,
            input_path=config.input_path,
            forecast_path=config.forecast_path)
    benchmark(mdl_name=config.mdl_name,
              forecast_path=config.forecast_path,
              score_path=config.score_path)
