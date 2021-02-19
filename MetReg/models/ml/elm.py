"""
from hpelm import ELM
"""
from MetReg.base.base_model import BaseModel
import numpy as np
np.random.seed(1)

from sklearn_extensions.extreme_learning_machines.elm import (ELMRegressor,
                                                              GenELMRegressor)
from sklearn_extensions.extreme_learning_machines.random_layer import \
    MLPRandomLayer, RandomLayer



class ExtremeLearningRegressor(BaseModel):

    def __init__(self,):
        self.regressor = None

    def fit(self, X, y):
        n_features = X.shape[-1]
        self.regressor = GenELMRegressor(hidden_layer=MLPRandomLayer(
          n_hidden=128, activation_func='sigmoid'))
        #mdl = ELM(30, 1)
        #mdl.add_neurons(16, 'sigm')
        #mdl.add_neurons(64, 'sigm')
        #mdl.add_neurons(32, 'sigm')
        #self.regressor = ELM(n_features, 1)
        #self.regressor.add_neurons(16, 'sigm')

        self.regressor.fit(X, y)
        return self
