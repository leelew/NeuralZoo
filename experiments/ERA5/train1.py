from MetReg.train.train_dl import get_callback
from MetReg.train.loss import MaskMSE, SSIM
from MetReg.utils.utils import _read_inputs, save2pickle
from MetReg.api.model_io import ModelInterface, ModelSaver
import os
import numpy as np
import argparse
import pickle
from sklearn.metrics import r2_score
import sys
sys.path.append('../../')


def main(mdl_name,
         input_path,
         model_path,
         forecast_path,
         batch_size,
         epochs,
         task=199):
    X_train0, X_valid0, y_train0, y_valid0, mask0 = _read_inputs(
        task=0,
        input_path=input_path,
        mask_path=input_path)
    X_train1, X_valid1, y_train1, y_valid1, mask1 = _read_inputs(
        task=1,
        input_path=input_path,
        mask_path=input_path)

    np.con



    save_path = forecast_path + mdl_name + \
        '/saved_model_' + str(task) + '.pickle'
    print(save_path)

    if os.path.exists(save_path):
        print('fuck')
    #    return 'already train'

    # read inputs
    X_train, X_valid, y_train, y_valid, mask = _read_inputs(
        task=task,
        input_path=input_path,
        mask_path=input_path)

    print(mask)
    # get shape
    N_t, T, H, W, F = X_train.shape
    N_v = X_valid.shape[0]

    """
    for i in range(H):
        for j in range(W):
            if np.isnan(mask[i, j]):
                y_train[:, 0, i, j, 0] = np.zeros((N_t,))
                y_valid[:, 0, i, j, 0] = np.zeros((N_v,))
    """
    # train & save model
    if mdl_name.split('.')[0] == 'sdl':

        mdl = ModelInterface(mdl_name=mdl_name).get_model()

        mdl.compile(optimizer='adam',  loss='mse')  # loss=MaskMSE(mask),)
        callbacks = get_callback(model_path, task)
        mdl.fit(X_train,
                y_train[:, 0, :, :, :],
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks)

        #mdl = ModelInterface(mdl_name=mdl_name).get_model()
        #mdl.load_weights(model_path + str(task) + '/' + str(task) + '.ckpt')
        # validation_split=0.1)
        # validation_split=0.2)
        y_pred_ = mdl.predict(X_valid)
        y_valid_ = y_valid[:, 0, :, :, 0]

        _, _, _, _, mask = _read_inputs(
            task=task,
            input_path=input_path,
            mask_path=input_path)
        print(mask)
        for i in range(H):
            for j in range(W):
                if not np.isnan(mask[i, j]):
                    print(r2_score(y_valid_[:, i, j], y_pred_[:, i, j]))

        ModelSaver(mdl, mdl_name=mdl_name,
                   dir_save=model_path + mdl_name,
                   name_save='/saved_model_' + str(task))()
    else:
        saved_mdl = [[] for i in range(H)]

        y_pred_ = np.full((N_v, H, W), np.nan)
        y_valid_ = np.full((N_v, H, W), np.nan)

        for i in range(H):
            for j in range(W):
                if not np.isnan(mask[i, j]):

                    mdl = ModelInterface(mdl_name=mdl_name).get_model()

                    if mdl_name.split('.')[0] == 'ml':

                        mdl.fit(X_train[:, :, i, j, :].reshape(N_t, -1),
                                y_train[:, 0, i, j, 0])
                        y_pred_[:, i, j] = np.squeeze(mdl.predict(
                            X_valid[:, :, i, j, :].reshape(N_v, -1)))
                        y_valid_[:, i, j] = np.squeeze(y_valid[:, 0, i, j, 0])

                        print(r2_score(y_valid_[:, i, j], y_pred_[:, i, j]))

                    elif mdl_name.split('.')[0] == 'dl':

                        if mdl_name.split('.')[1] in ['rnn', 'dnn']:

                            mdl.compile(optimizer='adam', loss='mse')
                            mdl.fit(X_train[:, :, i, j, :],
                                    y_train[:, 0, i, j, 0],
                                    batch_size=batch_size,
                                    epochs=epochs,)
                            y_pred_[:, i, j] = np.squeeze(
                                mdl.predict(X_valid[:, :, i, j, :]))
                            y_valid_[:, i, j] = np.squeeze(
                                y_valid[:, 0, i, j, 0])

                            print(
                                r2_score(y_valid_[:, i, j], y_pred_[:, i, j]))

                    saved_mdl[i].append(mdl)
                else:
                    saved_mdl[i].append(None)

        if mdl_name.split('.')[1] == 'cnn':
            mdl.compile(optimizer='adam', loss='mse')
            mdl.fit(X_train.reshape(N_t, H, W, F*T),
                    y_train[:, 0, :, :, 0],
                    batch_size=batch_size,
                    epochs=epochs,)
            y_pred_ = np.squeeze(
                mdl.predict(X_valid.reshape(N_v, H, W, F*T)))
            y_valid_ = np.squeeze(
                y_valid[:, 0, :, :, 0])
            for i in range(H):
                for j in range(W):
                    if not np.isnan(mask[i, j]):
                        print(r2_score(y_valid_[:, i, j], y_pred_[:, i, j]))

        try:
            ModelSaver(saved_mdl, mdl_name=mdl_name,
                       dir_save=model_path+mdl_name,
                       name_save='/saved_model_' + str(task))()
        except:
            print("Don't have saving model mode!")

        log = dict()
        log['y_pred'] = y_pred_
        log['y_valid'] = y_valid_

        save2pickle(log,
                    out_path=forecast_path + mdl_name,
                    out_file='/saved_model_' + str(task) + '.pickle')


if __name__ == "__main__":
    from parser import get_parse
    config = get_parse()

    main(mdl_name=config.mdl_name,
         input_path=config.input_path,
         model_path=config.model_path,
         forecast_path=config.forecast_path,
         batch_size=config.batch_size,
         epochs=config.epochs,
         task=config.num_jobs,
         )
