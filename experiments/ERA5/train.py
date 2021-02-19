import argparse
import pickle
import time
import sys
sys.path.append('../../')

import matplotlib.pyplot as plt
import numpy as np
from MetReg.api.model_io import ModelInterface, ModelSaver
from six.moves import cPickle
from sklearn.metrics import r2_score


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
    _, T, H, W, F = X_train.shape

    # train & save model
    if mdl_name.split('.')[0] == 'sdl':

        mdl = ModelInterface(mdl_name=mdl_name).get_model()

        mdl.compile(optimizer='adam', loss='mse')
        mdl.fit(X_train.reshape(-1, T, H, W, F),
                y_train[:, 0, :, :, :], batch_size=32, epochs=50)

        ModelSaver(mdl, mdl_name=mdl_name,
                   dir_save='/hard/lilu/saved_models/'+mdl_name,
                   name_save='/saved_model_' + str(task))()

    else:

        saved_mdl = [[] for i in range(H)]

        for i in range(H):
            for j in range(W):
                if not np.isnan(mask[i, j]):

                    mdl = ModelInterface(mdl_name=mdl_name).get_model()

                    if mdl_name.split('.')[0] == 'ml':

                        mdl.fit(X_train[:, :, i, j, :].reshape(
                                X_train.shape[0], -1),
                                y_train[:, 0, i, j, 0])

                    elif mdl_name.split('.')[0] == 'dl':

                        mdl.compile(optimizer='adam', loss='mse')
                        mdl.fit(X_train[:, :, i, j, :],
                                y_train[:, 0, i, j, 0],
                                batch_size=32, epochs=5,)

                    else:
                        print('manual train class')

                    saved_mdl[i].append(mdl)
                else:
                    saved_mdl[i].append(None)

        ModelSaver(saved_mdl, mdl_name=mdl_name,
                   dir_save='/hard/lilu/saved_models/'+mdl_name,
                   name_save='/saved_model_' + str(task))()


if __name__ == "__main__":

    parse = argparse.ArgumentParser()
    parse.add_argument('--mdl_name', type=str, default='dl.rnn.lstm')
    config = parse.parse_args()

    main(mdl_name=config.mdl_name,
         input_path='/WORK/sysu_yjdai_6/hard/lilu/ERA5_1981_2017_DD_A1/',
         mask_path='/WORK/sysu_yjdai_6/hard/lilu/ERA5_1981_2017_DD_A1/',
         task=0)

    """
    for task in range(200):
        print('task = {}'.format(task))

        main(mdl_name=config.mdl_name, task=task)
    """
