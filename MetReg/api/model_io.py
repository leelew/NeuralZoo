import os
import pickle

from MetReg.benchmark.benchmark import _benchmark_array, _benchmark_img
from MetReg.models.dl.lstm import bilstm, gru, lstm, rnn
from MetReg.models.ml.automl import automl
from MetReg.models.ml.elm import elm
from MetReg.models.ml.gaussian import GP
from MetReg.models.ml.linear import LR, ElasticNet, Lasso, Ridge
from MetReg.models.ml.svr import svr
from MetReg.models.ml.tree import DT, GBDT, RF, LightGBM, Xgboost
from MetReg.train.train_ml import train_ml
from MetReg.utils.parser import get_lr_args


class model_generator:
    """generate model according model name."""

    def __init__(self, mdl_name):
        self.mdl_name = mdl_name

        self.mdl_type = mdl_name.split('.')[0]

    def __call__(self):
        if 'lr' in self.mdl_name.lower():
            mdl = self._get_lr_mdl(self.mdl_name)
        elif 'tree' in self.mdl_name.lower():
            mdl = self._get_tree_mdl(self.mdl_name)
        elif 'rnn' in self.mdl_name.lower():
            mdl = self._get_rnn_mdl(self.mdl_name)
        elif 'gp' in self.mdl_name.lower():
            mdl = self._get_gaussian_mdl(self.mdl_name)
        elif 'svr' in self.mdl_name.lower():
            mdl = self._get_svr_mdl(self.mdl_name)
        elif 'elm' in self.mdl_name.lower():
            mdl = self._get_elm_mdl(self.mdl_name)
        elif 'auto' in self.mdl_name.lower():
            mdl = self._get_automl_mdl(self.mdl_name)
        else:
            raise NameError('Have not support this model!')
        return mdl

    def _get_elm_mdl(self, mdl_name):
        elm_hash = {
            'elm': elm()()
        }
        return elm_hash[mdl_name.split('.')[-1]]

    def _get_svr_mdl(self, mdl_name):
        svr_hash = {
            'svr': svr()()
        }
        return svr_hash[mdl_name.split('.')[-1]]

    def _get_gaussian_mdl(self, mdl_name):
        gaussian_hash = {
            'gpr': GP()()
        }
        return gaussian_hash[mdl_name.split('.')[-1]]

    def _get_lr_mdl(self, mdl_name):
        config = get_lr_args()

        lr_hash = {
            'lr': LR(config)(),
            'ridge': Ridge(config)(),
            'lasso': Lasso(config)(),
            'elasticnet': ElasticNet(config)(),
        }
        return lr_hash[mdl_name.split('.')[-1]]

    def _get_tree_mdl(self, mdl_name):

        tree_hash = {
            'dt': DT()(),
            'rf': RF()(),
            'gbdt': GBDT()(),
            'xgboost': Xgboost()(),
            'lightgbm': LightGBM()(),
        }
        return tree_hash[mdl_name.split('.')[-1]]

    def _get_rnn_mdl(self, mdl_name):

        rnn_hash = {
            'lstm': lstm()(),
            'rnn': rnn()(),
            'gru': gru()(),
            'bilstm': bilstm()(),
        }
        return rnn_hash[mdl_name.split('.')[-1]]

    def _get_automl_mdl(self, mdl_name):

        automl_hash = {
            'ml': automl()
        }


class model_benchmarker:

    def __init__(self,
                 mdl,
                 X,
                 y=None):
        self.mdl = mdl
        self.X = X
        self.y = y

    def __call__(self):
        y_pred = self.mdl.predict(self.X)
        return _benchmark_array(self.X, y_pred)()


class model_loader:
    def __init__(self, save_path): pass

    def __call__(self): pass


class model_saver:

    def __init__(self, mdl, dir_save, name_save):
        self.mdl = mdl
        self.dir_save = dir_save
        self.name_save = name_save

    def __call__(self):
        if not os.path.isdir(self.dir_save):
            os.mkdir(self.dir_save)

        pickle.dump(self.mdl, open(
            self.dir_save+self.name_save, 'wb'))
