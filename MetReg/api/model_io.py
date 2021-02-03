import os
import pickle

from MetReg.benchmark.benchmark import _benchmark_array, _benchmark_img
from MetReg.models.ml.elm import ExtremeLearningRegressor
from MetReg.models.ml.gp import GaussianProcessRegressor
from MetReg.models.ml.knn import KNNRegressor
from MetReg.models.ml.linear import (BaseLinearRegressor, ElasticRegressor,
                                     ExpandLinearRegressor, LassoRegressor,
                                     RidgeRegressor)
from MetReg.models.ml.mlp import MLPRegressor
from MetReg.models.ml.svr import LinearSVRegressor, SVRegressor
from MetReg.models.ml.tree import (AdaptiveBoostingRegressor,
                                   BaseTreeRegressor, ExtraTreesRegressor,
                                   ExtremeGradientBoostingRegressor,
                                   GradientBoostingRegressor,
                                   LightGradientBoostingRegressor,
                                   RandomForestRegressor)

from MetReg.models.dl.rnn import BaseRNNRegressor


class ModelInterface():
    """generate model according model name."""

    def __init__(self,
                 mdl_name,
                 params=None):
        self.mdl_general = mdl_name.split('.')[0]  # ml/dl
        self.mdl_type = mdl_name.split('.')[1]
        self.mdl_name = mdl_name.split('.')[2]

        self.params = params  # TODO: configuration parser

    def get_model(self):

        if 'linear' in self.mdl_type.lower():
            mdl = self._get_lr_mdl(self.mdl_name)
        elif 'tree' in self.mdl_type.lower():
            mdl = self._get_tree_mdl(self.mdl_name)
        elif 'svr' in self.mdl_type.lower():
            mdl = self._get_svr_mdl(self.mdl_name)
        elif 'gp' in self.mdl_type.lower():
            mdl = self._get_gp_mdl(self.mdl_name)
        elif 'mlp' in self.mdl_type.lower():
            mdl = self._get_mlp_mdl(self.mdl_name)
        elif 'elm' in self.mdl_type.lower():
            mdl = self._get_elm_mdl(self.mdl_name)
        elif 'knn' in self.mdl_type.lower():
            mdl = self._get_knn_mdl(self.mdl_name)

        elif 'rnn' in self.mdl_type.lower():
            mdl = self._get_rnn_mdl(self.mdl_name)

        else:
            raise NameError('Have not support this model!')
        return mdl

    def _get_lr_mdl(self, mdl_name):
        lr_hash = {
            'base': BaseLinearRegressor(),
            'ridge': RidgeRegressor(),
            'lasso': LassoRegressor(),
            'elastic': ElasticRegressor(),
        }
        return lr_hash[mdl_name]

    def _get_tree_mdl(self, mdl_name):
        tree_hash = {
            'base': BaseTreeRegressor(),
            'rf': RandomForestRegressor(),
            'etr': ExtraTreesRegressor(),
            'adaboost': AdaptiveBoostingRegressor(),
            'gbdt': GradientBoostingRegressor(),
            'xgboost': ExtremeGradientBoostingRegressor(),
            'lightgbm': LightGradientBoostingRegressor(),
        }
        return tree_hash[mdl_name]

    def _get_svr_mdl(self, mdl_name):
        svr_hash = {
            'svm': SVRegressor(),
            'linear': LinearSVRegressor()
        }
        return svr_hash[mdl_name]

    def _get_gp_mdl(self, mdl_name):
        gp_hash = {
            'gp': GaussianProcessRegressor()
        }
        return gp_hash[mdl_name]

    def _get_mlp_mdl(self, mdl_name):
        mlp_hash = {
            'mlp': MLPRegressor()
        }
        return mlp_hash[mdl_name]

    def _get_knn_mdl(self, mdl_name):
        knn_hash = {
            'knn': KNNRegressor(),
        }
        return knn_hash[mdl_name]

    def _get_elm_mdl(self, mdl_name):
        elm_hash = {
            'elm': ExtremeLearningRegressor()
        }
        return elm_hash[mdl_name]

    def _get_rnn_mdl(self, mdl_name):
        rnn_hash = {
            'base': BaseRNNRegressor()
        }
        return rnn_hash[mdl_name]


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
