import datetime as dt
import math
import os
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from lightgbm.sklearn import LGBMRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
#from keract import get_activations
from tensorflow.keras import Input, Model, models
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import (LSTM, Activation, BatchNormalization,
                                     Conv2D, Conv3D, ConvLSTM2D, Dense,
                                     Dropout, Lambda, concatenate, dot)

from attention import CBAM_block
from data_loader import gen_esa_x_y, gen_flx_x_y, gen_smap_x_y
from utils import gen_params, gen_test_data, r2, tictoc


def compare_models_flx(file_path,
                       time_resolution='DD',
                       begin_date='2019-05-03',
                       end_date='2109-05-03'):

    log = dict()

    # ML
    x_train, x_valid, y_train, y_valid = gen_flx_x_y(file_path, batch=False)

    model = ML_Models(x_train, x_valid, y_train, y_valid)

    log['Ridge'] = model.Ridge()
    log['SVR'] = model.SVR()
    log['RF'] = model.RF()
    log['XGB'] = model.XGB()
    log['LGB'] = model.LGB()
    #log['stacking'] = model.stacking()

    # DL
    x_train, x_valid, y_train, y_valid = gen_flx_x_y(file_path, batch=True)

    model = DL_Models(x_train, x_valid, y_train, y_valid, DNN=True)
    model.DNN()
    log['DNN'] = model.train()

    model = DL_Models(x_train, x_valid, y_train, y_valid, LSTM=True)
    model.LSTM()
    log['LSTM'] = model.train()

    return log


def compare_models_esa(df,
                       time_resolution='DD',
                       begin_date='2002-06-19',
                       end_date='2011-06-19'):

    log = dict()

    # ML
    x_train, x_valid, y_train, y_valid = gen_esa_x_y(df, batch=False)

    model = ML_Models(x_train, x_valid, y_train, y_valid)

    log['Ridge'] = model.Ridge()
    log['SVR'] = model.SVR()
    log['RF'] = model.RF()
    log['XGB'] = model.XGB()
    log['LGB'] = model.LGB()
    #log['stacking'] = model.stacking()

    # DL
    x_train, x_valid, y_train, y_valid = gen_esa_x_y(df, batch=True)

    model = DL_Models(x_train, x_valid, y_train, y_valid, DNN=True)
    model.DNN()
    log['DNN'] = model.train()

    model = DL_Models(x_train, x_valid, y_train, y_valid, DNN=False)
    model.LSTM()
    log['LSTM'] = model.train()

    return log


def compare_models_smap(df,
                        time_resolution='3HH',
                        begin_date='2015-03-31',
                        end_date='2019-01-24'):

    log = dict()

    # ML
    x_train, x_valid, y_train, y_valid = gen_smap_x_y(df, batch=False)

    model = ML_Models(x_train, x_valid, y_train, y_valid)

    log['Ridge'] = model.Ridge()
    log['SVR'] = model.SVR()
    log['RF'] = model.RF()
    log['XGB'] = model.XGB()
    log['LGB'] = model.LGB()
    #log['stacking'] = model.stacking()

    # DL
    x_train, x_valid, y_train, y_valid = gen_smap_x_y(df, batch=True)

    model = DL_Models(x_train, x_valid, y_train, y_valid, DNN=True)
    model.DNN()
    log['DNN'] = model.train()

    model = DL_Models(x_train, x_valid, y_train, y_valid, LSTM=True)
    model.LSTM()
    log['LSTM'] = model.train()

    return log


class ML_Models():

    def __init__(self, x_train, x_valid, y_train, y_valid, cv=False):

        self.x_train = x_train
        self.x_valid = x_valid
        self.y_train = y_train
        self.y_valid = y_valid
        self.cv = cv
        self.importance = []
        self.num_feature = np.shape(x_train)[-1]

    def print_log(self):

        self._metrics()

        print("train set R2 is {}".format(self.train_metrics))
        print("valid set R2 is {}".format(self.valid_metrics))

    def gen_log(self):

        self.log = dict()
        self.log['train_metrics'] = self.train_metrics
        self.log['valid_metrics'] = self.valid_metrics
        self.log['pred_train'] = self.pred_train
        self.log['pred_valid'] = self.pred_valid
        self.log['x_train'] = self.x_train
        self.log['x_valid'] = self.x_valid
        self.log['y_train'] = self.y_train
        self.log['y_valid'] = self.y_valid
        self.log['importance'] = self.importance

    def save_model(self):
        pass
        # save model
        # LGB.save_model('model.txt')

        # load model
        # LGB = lgb.Booster(model_file='txt')

    def _metrics(self):

        try:
            self.train_metrics = r2_score(np.squeeze(
                self.y_train), np.squeeze(self.pred_train))
            self.valid_metrics = r2_score(np.squeeze(
                self.y_valid), np.squeeze(self.pred_valid))
        except:
            self.train_metrics = []
            self.valid_metrics = []

    @tictoc
    def Ridge(self, alpha=.1):

        LR = Ridge(alpha=alpha)
        LR.fit(self.x_train, self.y_train)

        self.pred_train = LR.predict(self.x_train)
        self.pred_valid = LR.predict(self.x_valid)

        self.print_log()
        self.gen_log()

        return self.log

    @tictoc
    def SVR(self):

        SVR = svm.SVR()
        SVR.fit(self.x_train, self.y_train)

        self.pred_train = SVR.predict(self.x_train)
        self.pred_valid = SVR.predict(self.x_valid)

        self.print_log()
        self.gen_log()

        return self.log

    @tictoc
    def RF(self):

        RF = RandomForestRegressor()
        RF.fit(self.x_train, self.y_train)

        self.pred_train = RF.predict(self.x_train)
        self.pred_valid = RF.predict(self.x_valid)

        self.print_log()
        self.gen_log()

        return self.log

    @tictoc
    def XGB(self):

        trainDMat = xgb.DMatrix(data=self.x_train, label=self.y_train)
        testDMat = xgb.DMatrix(data=self.x_valid, label=self.y_valid)

        parameters = gen_params()

        XGB = xgb.train(
            params=parameters,
            dtrain=trainDMat,
            evals=[(trainDMat, 'train'),
                   (testDMat, 'eval')]
        )

        self.pred_train = XGB.predict(trainDMat)
        self.pred_valid = XGB.predict(testDMat)

        self.print_log()
        self.gen_log()

        return self.log

    @tictoc
    def LGB(self):

        # NOTE: It should be list, numpy 1-D array or pandas Series
        trainSet = lgb.Dataset(
            data=self.x_train, label=np.squeeze(self.y_train))
        validSet = lgb.Dataset(
            data=self.x_valid, label=np.squeeze(self.y_valid))

        parameters = gen_params()

        LGB = lgb.train(
            params=parameters,
            train_set=trainSet,
            valid_sets=validSet
        )

        self.importance = LGB.feature_importance()
        print(self.importance)
        self.pred_train = LGB.predict(self.x_train)
        self.pred_valid = LGB.predict(self.x_valid)

        self.print_log()
        self.gen_log()

        return self.log

    @tictoc
    def stacking(self):

        log_RF = self.RF()
        log_LGB = self.LGB()
        log_XGB = self.XGB()

        train = np.concatenate(
            (log_RF['pred_train'].reshape(-1, 1),
             log_XGB['pred_train'].reshape(-1, 1),
             log_LGB['pred_train'].reshape(-1, 1)), axis=1)

        valid = np.concatenate(
            (log_RF['pred_valid'].reshape(-1, 1),
             log_XGB['pred_valid'].reshape(-1, 1),
             log_LGB['pred_valid'].reshape(-1, 1)), axis=1)

        MLP = MLPRegressor()
        MLP.fit(train, self.y_train)

        self.pred_train = MLP.predict(train)
        self.pred_valid = MLP.predict(valid)

        self.print_log()
        self.gen_log()

        return self.log


class DL_Models(ML_Models):

    def __init__(self, x_train, x_valid, y_train, y_valid,
                 logdir=[],
                 learning_rate=0.01,
                 epoch=5,
                 loss='mse',
                 metrics=['mse', 'mae'],
                 DNN=False,
                 LSTM=False,
                 ConvLSTM=False):

        self.learning_rate = learning_rate
        self.epoch = epoch
        self.loss = loss
        self.metrics = metrics  # metric for judge model performance
        self.importance = []
        self.logdir = logdir

        if x_train.ndim == 3:
            self.train_batch_size, self.timesteps, self.num_feature = np.shape(
                x_train)
            self.valid_batch_size, self.timesteps, self.num_feature = np.shape(
                x_valid)
        elif x_train.ndim == 5:
            self.train_batch_size, self.timesteps, _, _, self.num_feature = np.shape(
                x_train)
            self.valid_batch_size, self.timesteps, _, _, self.num_feature = np.shape(
                x_valid)

        self.x_train = x_train
        self.x_valid = x_valid
        self.y_train = y_train
        self.y_valid = y_valid

        if DNN:
            self.x_train = x_train.reshape(self.train_batch_size, -1)
            self.x_valid = x_valid.reshape(self.valid_batch_size, -1)
            self.y_train = y_train.reshape(self.train_batch_size, -1)
            self.y_valid = y_valid.reshape(self.valid_batch_size, -1)
        elif LSTM:
            self.y_train = np.squeeze(y_train)
            self.y_valid = np.squeeze(y_valid)

            print(self.x_train.shape)
            print(self.y_train.shape)

    def DNN(self):

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(
            128, activation='relu', input_shape=[self.timesteps*self.num_feature]))
        self.model.add(tf.keras.layers.Dense(256, activation='relu'))
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1))

        self.model.summary()

    def LSTM(self):

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.LSTM(
            units=64, activation='tanh',
            return_sequences=True, input_shape=[None, self.num_feature]))

        self.model.add(tf.keras.layers.LSTM(units=1, activation='tanh'))

        self.model.summary()

    def convLSTM(self):

        # train as shape 5-dims: []
        self.model = tf.keras.models.Sequential()
        self.model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                                  input_shape=(None, 30, 30, 1), padding='same', return_sequences=True,
                                  activation='tanh', recurrent_activation='hard_sigmoid',
                                  kernel_initializer='glorot_uniform', unit_forget_bias=True,
                                  dropout=0.3, recurrent_dropout=0.3, go_backwards=True))
        self.model.add(BatchNormalization())

        self.model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True,
                                  activation='tanh', recurrent_activation='hard_sigmoid',
                                  kernel_initializer='glorot_uniform', unit_forget_bias=True,
                                  dropout=0.4, recurrent_dropout=0.3, go_backwards=True))
        self.model.add(BatchNormalization())

        self.model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True,
                                  activation='tanh', recurrent_activation='hard_sigmoid',
                                  kernel_initializer='glorot_uniform', unit_forget_bias=True,
                                  dropout=0.4, recurrent_dropout=0.3, go_backwards=True))
        self.model.add(BatchNormalization())

        self.model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False,
                                  activation='tanh', recurrent_activation='hard_sigmoid',
                                  kernel_initializer='glorot_uniform', unit_forget_bias=True,
                                  dropout=0.4, recurrent_dropout=0.3, go_backwards=True))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(filters=1, kernel_size=(1, 1),
                              activation='sigmoid',
                              padding='same', data_format='channels_last'))

        self.model.summary()

    def att_convLSTM(self):

        i = Input(shape=(100, 10, 10, 1))

        x = CBAM_block(i)
        x = ConvLSTM2D(filters=64, kernel_size=(3, 3),
                       padding='same', return_sequences=True,
                       activation='tanh', recurrent_activation='hard_sigmoid',
                       kernel_initializer='glorot_uniform', unit_forget_bias=True,
                       dropout=0.3, recurrent_dropout=0.3, go_backwards=True)(x)
        x = BatchNormalization()(x)
        x = ConvLSTM2D(filters=32, kernel_size=(3, 3),
                       padding='same', return_sequences=False,
                       activation='tanh', recurrent_activation='hard_sigmoid',
                       kernel_initializer='glorot_uniform', unit_forget_bias=True,
                       dropout=0.3, recurrent_dropout=0.3, go_backwards=True)(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=1, kernel_size=(1, 1),
                   activation='sigmoid',
                   padding='same', data_format='channels_last')(x)

        self.model = Model(inputs=i, outputs=x)
        self.model.summary()

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

    @tictoc
    def train(self):

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss=self.loss,
            metrics=self.metrics
        )

        self.history = self.model.fit(self.x_train, self.y_train,
                                      epochs=self.epoch,
                                      validation_split=0.3)

        self.pred_train = self.model.predict(self.x_train)
        self.pred_valid = self.model.predict(self.x_valid)

        self.print_log()
        self.gen_log()

        return self.log


if __name__ == "__main__":

    # generate test data for model testing
    x_train, x_valid, y_train, y_valid = gen_test_data(ML=True)

    # class model
    model = ML_Models(x_train, x_valid, y_train, y_valid, cv=False)
    model = DL_Models(x_train, x_valid, y_train, y_valid,
                      learning_rate=0.01,
                      epoch=1,
                      loss='mse',
                      metrics=['mse', 'mae'])

    #########################
    # ML MODELS
    #########################
    # ridge regression
    # model.Ridge()
    # support vector machine
    # model.SVR()
    # random forest
    # model.RF()
    # xgboost
    # model.XGB()
    # lightgbm
    # model.LGB()

    #########################
    # DL MODELS
    #########################
    # LSTM
    # model.LSTM()
    # DNN
    # model.DNN()

    # model.train()

"""
    if cv:
        # cv mode
        min_error = float('Inf')
        best_params = {}

        for max_depth in range(2, 7, 1):
            for min_data_in_leaf in [50, 100]:
                for num_leaves in [4, 8, 16, 32]:

                    print("\nHyperparams is \nmax_depth : {} \nnum_leaves : {} \nmin_data_in_leaf : {}".format(
                        max_depth, num_leaves, min_data_in_leaf))

                    # params['max_bin'] = max_bin
                    params['num_leaves'] = num_leaves
                    params['max_depth'] = max_depth
                    params['min_data_in_leaf'] = min_data_in_leaf

                    cv_results = lgb.cv(
                        params,
                        trainSet,
                        nfold=3,
                        metrics=['rmse'],
                        stratified=False,
                        shuffle=False
                    )

                    _min_error_1step = pd.Series(cv_results['rmse-mean']).min()

                    if _min_error_1step < min_error:
                        min_error = _min_error_1step
                        best_params['num_leaves'] = num_leaves
                        # best_params['max_bin'] = max_bin
                        best_params['max_depth'] = max_depth
                        best_params['min_data_in_leaf'] = min_data_in_leaf

        # params['max_bin'] = best_params['max_bin']
        params['num_leaves'] = best_params['num_leaves']
        params['max_depth'] = best_params['max_depth']
        params['min_data_in_leaf'] = best_params['min_data_in_leaf']

        
        # hyper tuning using sklearn api
        hyper_params = {
            'max_depth': [4, 5, 8, -1],
            'num_leaves': [15, 20, 25, 30],
            'learning_rate': [0.1, 0.01, 0.02],
        }

        est = lgb.LGBMRegressor()
        gs = GridSearchCV(est, hyper_params, scoring='r2', cv=5, verbose=1)

        gs_result = gs.fit(x_train_scaled, y_train)
        
"""
