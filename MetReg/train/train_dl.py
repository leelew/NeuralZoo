# -----------------------------------------------------------------------------
#                                Train module                                 #
# -----------------------------------------------------------------------------
# author: Lu Li                                                               #
# mail: lilu35@mail2.sysu.edu.cn                                              #
# -----------------------------------------------------------------------------
# This Repo is train module of artifical intelligence models, it has three    #
# types class, train class for machine learning models using skcit-learn libs,#
# deep learning models using keras libs in tensorflow (high-level API) and    #
# other models using gradientTape def in tensorflow libs. Notability, all     #
# classes are limited interface based on abstractmethods class in base.py     #
# -----------------------------------------------------------------------------


import os
import time

import numpy as np

from base import Base_train_ml, Base_train_keras
from utils import get_ml_models, tictoc
import xgboost as xgb
import lightgbm as lgb

from models.ConvRNN.main import trajGRU
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from metrics import Metrics
import tensorflow as tf

class train_ml(Base_train_ml):

    def __init__(self, x_train, y_train,
                 model_name: list = ['linear.default'],):
        """train for machine learing based on sklearn. More info
        pls see the annotation of Base_train_ml class in base.py.

        Args:
            model_name (list):
                model name list of target models which must
                obey 'class.def' type, the standard is shown
                in guide.pdf

        Raises:
            KeyError: module could raise key error if don't 
                      have model name or have wrong model name. 
        """

        if model_name is None:
            raise KeyError('please select ML models!')
        else:
            # generate regression list by model name list
            self.models = get_ml_models(model_name)
            self.mdl_name = model_name

        # define train mode
        self._train_mode = None
        # check data & set train mode
        self._check_data_get_mode(x_train, y_train)
        # get attribute of data
        self._get_data_attr(x_train, y_train)

    def _check_data_get_mode(self, x_train=None, y_train=None):

        if x_train is None:
            raise KeyError('pls give an input!')
        elif x_train.ndim != 2 and x_train.ndim != 4:
            raise KeyError('pls give a 2D OR 4D input!')
        elif x_train.ndim == 2:
            print('[Global] Now going into global mode!')
            self._train_mode = 'global'
        else:
            print('[Grid] Now going into grid mode!')
            self._train_mode = 'grid'

    def _get_data_attr(self, x_train, y_train):

        if self._train_mode == 'global':
            self.train_timestep, self.Nfeature = x_train.shape
        elif self._train_mode == 'grid':
            self.train_timestep, self.Nlat, self.Nlon, self.Nfeature = \
                x_train.shape
        else:
            raise KeyError('please give a train mode!')

    @staticmethod
    def _train_grid(reg=None, x_train=None, y_train=None):
        """train machine learning models by sklearn on single grid.

        Returns:
            reg (class): 
                trained regression class could directly use ().fit.
        """
        if reg is None:
            raise KeyError('please give a model class')
        else:
            # train

            reg.fit(x_train, y_train)

        return reg

    def global_train_models(self, x_train, y_train=None) -> dict:
        """[global] mode training.

        Returns:
            global_model (dict):
                list shape of (number of models, ) that contain trained 
                regression class with different models. 
        """
        if self._train_mode == 'grid':
            raise KeyError('pls use grid_train_models func instead !')
        elif self._train_mode == 'global':
            # init
            global_model = {}
            # loop for models class
            for index, mdl in enumerate(self.models):

                a = time.time()
                reg = self._train_grid(mdl, x_train, y_train)
                global_model[self.mdl_name[index]] = reg
                b = time.time()
                print('finish training {} model using {}s'.format(
                    self.mdl_name[index], b-a))
        else:
            raise KeyError('pls give a train mode!')

        return global_model

    def _train_grids(self, reg, x_train, y_train, mdl_name) -> list:
        """train machine learning models on lat-lon grids of single model.

        Args:
            reg (sklearn models): 
            x_train (np array):
                numpy array of shape (timestep, height, width, channel)
            y_train (np array): 
                numpy array of shape (timestep, height, width, )

        Returns:
            reg_list (list):
                2D nest list contain models of size (height,width).
        """
        if self._train_mode == 'grid':
            # init 2D list
            reg_list = [[] for i in range(self.Nlat)]
            # append trained reg
            for i in range(self.Nlat):
                for j in range(self.Nlon):

                    """
                    if 'Xgboost' in mdl_name:
                        train = xgb.DMatrix(x_train[:,i,j,:], label=y_train[:,i,j])
                        _reg = train_ml._train_grid(reg, train)
                        reg_list[i].append(_reg)

                    elif 'LightGBM' in mdl_name:
                        train = lgb.Dataset(x_train[:,i,j,:], label=y_train[:,i,j])
                        _reg = train_ml._train_grid(reg, train)
                        reg_list[i].append(_reg)
                    """

                    _reg = train_ml._train_grid(reg,
                                                x_train[:, i, j, :],
                                                y_train[:, i, j])
                    reg_list[i].append(_reg)
        else:
            raise KeyError('pls give correct train mode!')

        return reg_list

    def grid_train_models(self, x_train, y_train) -> dict:

        """[grid] mode training.

        Returns:
            grid_model (dict):
                Nest dict of shape (num_models), the key of dict is 
                is model name from model_name list, the corresponding layers 
                is trained regression list of size (height, width) generated
                from _train_grids() def.
        """
        a = time.time()
        if self._train_mode == 'global':
            raise KeyError('pls used global_train_models instead!')
        elif self._train_mode == 'grid':
            # init
            grid_model = {}
            # loop for models class
            for index, mdl in enumerate(self.models):

                reg_list = self._train_grids(mdl, x_train, y_train, self.mdl_name[index])
                grid_model[self.mdl_name[index]] = reg_list
        else:
            raise KeyError('pls give a model !')
        b = time.time()
        print(b-a)
        return grid_model


class train_keras(Base_train_keras):

    """train deep learning model based on keras API. This class only used for
    global training of DL models (include CNNs, RNNs, ConvRNNs, GNN). GAN and 
    some own-build DL models pls refer to train_tf class. further info pls 
    see guide.pdf. 

    Args:
        Base_train_keras (base class):
            basic annonation refer to base class.
    """

    def __init__(self, x_train, y_train, mdl, config=None, mode='global'):
        self.mode = mode
        self._get_data_attr(x_train, y_train)

    def _check_data_get_mode(self, x_train, y_train):

        if x_train.ndim not in [4, 5]:
            raise KeyError('pls give a correct data for keras training!')

    def _get_data_attr(self, x_train, y_train):

        if self.mode == 'global' and x_train.ndim == 4:
            print('CNN global training')
            self.S, self.H, self.W, self.F = x_train.shape
        elif self.mode == 'global' and x_train.ndim == 5:
            print('ConvRNN global training')
            self.S, self.T, self.H, self.W, self.F = x_train.shape
        elif self.mode == 'grid' and x_train.ndim == 5:
            print('RNN grids training')
            self.S, self.T, self.H, self.W, self.F = x_train.shape
        elif self.mode == 'global' and x_train.ndim == 3:
            print('RNN global training')
            self.S, self.T, self.F = x_train.shape
        else:
            raise KeyError('pls set a training mode!')

    def train_global(self,
                    mdl, x_train, y_train,
                    optimizer='adam',
                    loss='mse',
                    batch_size=32,
                    epochs=10,
                    validation_split=0.2):

        if self.mode == 'grid':
            raise KeyError('pls refer to train_grid() func')
        else:
            mdl.compile(
                optimizer=optimizer,
                loss=loss,
                metrics='mse')

            mdl.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                )#validation_split=validation_split)

        return mdl

    def train_grid(self, mdl, x_train, y_train,                    
                   optimizer='adam',
                   loss='mse',
                   batch_size=32,
                   epochs=50,
                   validation_split=0.2):

        if self.mode == 'global':
            raise KeyError('pls refer to train_global() func')
        else:
            models = [[] for i in range(self.H)]

            for i in range(self.H):
                for j in range(self.W):

                    mdl.compile(optimizer=optimizer,
                                loss=loss,
                                metrics='mse')

                    mdl.fit(
                        x_train[:,:,i,j,:], y_train[:,i,j],
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=validation_split)
                    
                    models[i].append(mdl)
            
        return models

class train_trajGRU():

    def __init__(self, epochs=1):

        self.mdl = trajGRU()
        self.loss = MeanSquaredError()
        self.optimizer = Adam()

        self.epochs = epochs

    def train(self, x_train, y_train):

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.batch(1000)

        for i in range(self.epochs):
            for x, y in train_dataset:
                
                with tf.GradientTape() as tape:
                    x = tf.keras.backend.permute_dimensions(x,(1,0,2,3,4))

                    prediction = self.mdl(x)
                    loss = self.loss(prediction, y)
                    print(loss)
                
                gradients = tape.gradient(loss, self.mdl.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.mdl.trainable_variables))

    def evaluate(self, x_valid, y_valid):

        y_predict = self.mdl(x_valid).numpy()
        print(y_predict.shape)
        Metrics(y_valid, y_predict).get_sklearn_metrics(y_valid, y_predict)

class train_gan():

    def __init__(self): pass

    @tf.function
    def train_step(self, gen, disc, optimizer, x_train, y_train): pass
                
    def train(self, x_train, y_train): pass




class Trainer(object):

    def __init__(self, 
                model,
                loss_func,
                optimizer,
                logger,
                device,
                ):
        pass

    def prepare(self): pass

    def to(self, device): pass

    


         


