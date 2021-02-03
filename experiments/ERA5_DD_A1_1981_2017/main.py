import argparse
import pickle
from six.moves import cPickle
import matplotlib.pyplot as plt
import numpy as np
from MetReg.api.model_io import (ModelInterface,
                                 model_saver,)
from sklearn.metrics import r2_score
import json
import time


def _read_inputs(task,
                 input_path='/hard/lilu/ERA5_1981_2017_DD_A1/',
                 mask_path='/hard/lilu/ERA5_1981_2017_DD_A1/',):
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


def main(mdl_name='ml.tree.lightgbm',
         input_path='/hard/lilu/ERA5_1981_2017_DD_A1/',
         mask_path='/hard/lilu/ERA5_1981_2017_DD_A1/',
         task=199,):

    # read inputs
    X_train, X_valid, y_train, y_valid, mask = _read_inputs(task=task)

    # get shape
    NLAT, NLON = mask.shape

    # saved model
    saved_mdl = [[] for i in range(NLAT)]

    a = time.time()
    for i in range(NLAT):
        for j in range(NLON):
            if not np.isnan(mask[i, j]):
                # training & saving trained-models
                mdl = ModelInterface(mdl_name=mdl_name).get_model()
                if mdl_name.split('.')[0] == 'ml':

                    mdl.fit(
                        X_train[:, :, i, j, :].reshape(
                            X_train.shape[0], -1),
                        y_train[:, 0, i, j, 0])

                    y_predict = mdl.predict(
                        X_valid[:, :, i, j, :].reshape(X_valid.shape[0], -1))
                    # print(y_predict)
                    #print(y_valid[:, 0, i, j, 0])
                    print(r2_score(y_valid[:, 0, i, j, 0], y_predict))

                elif mdl_name.split('.')[0] == 'dl':
                    mdl.compile(optimizer='adam', loss='mse')
                    mdl.fit(
                        X_train[:, :, i, j, :],
                        y_train[:, 0, i, j, 0], batch_size=32, epochs=5,)

                    y_predict = mdl.predict(X_valid[:, :, i, j, :])
                    print(r2_score(y_valid[:, 0, i, j, 0], y_predict))
                else:
                    print('manual train class')

                saved_mdl[i].append(mdl)
            else:
                saved_mdl[i].append(None)
    b = time.time()
    print(b-a)
    try:
        model_saver(saved_mdl,
                    dir_save='/hard/lilu/saved_models/'+mdl_name,
                    name_save='/saved_model_' + str(task) + '.pickle')()
    except:
        from hpelm import ELM
        # ELM().save


if __name__ == "__main__":

    parse = argparse.ArgumentParser()
    parse.add_argument('--mdl_name', type=str, default='ml.lr.ridge')
    config = parse.parse_args()

    main(mdl_name=config.mdl_name, task=0)
    """
    for task in range(200):
        print('task = {}'.format(task))

        main(mdl_name=config.mdl_name, task=task)
    """