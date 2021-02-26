import argparse
import pickle
import sys
sys.path.append('../../')

import numpy as np
import tensorflow as tf
from MetReg.benchmark.benchmark import ScoreBoard
from MetReg.utils.utils import (_get_task_from_regions, _read_inputs,
                                save2pickle)

def ensemble(): pass


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
        try:
            X_train, X_valid, y_train, y_valid, mask = _read_inputs(
                task=num_jobs, input_path=input_path, mask_path=input_path)

            y_pred_, y_true_ = process(
                mdl_name=mdl_name, forecast_path=forecast_path,
                task=num_jobs, threshold=0)
            y_pred[:, attr[0]:attr[0]+18, attr[1]:attr[1]+18] = y_pred_ + \
                mask.reshape(1, 18, 18).repeat(N, axis=0)
            y_true[:, attr[0]:attr[0]+18, attr[1]:attr[1]+18] = y_true_ + \
                mask.reshape(1, 18, 18).repeat(N, axis=0)
        except:
            print("{} job is not trained".format(num_jobs))

    save2pickle(y_pred,
                out_path=forecast_path,
                out_file=mdl_name.split('.')[-1] + '_pred.pickle')
    save2pickle(y_true,
                out_path=forecast_path,
                out_file=mdl_name.split('.')[-1] + '_true.pickle')


def benchmark(mdl_name,
              forecast_path,
              score_path):

    f = open(forecast_path+mdl_name.split('.')[-1]+'_pred.pickle', 'rb')
    y_pred = pickle.load(f)
    f = open(forecast_path+mdl_name.split('.')[-1]+'_true.pickle', 'rb')
    y_true = pickle.load(f)

    sb = ScoreBoard(mode=-1)
    score = sb.benchmark(y_true, y_pred)

    save2pickle(score,
                out_path=score_path,
                out_file=mdl_name.split('.')[-1] + '_score.pickle')

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
