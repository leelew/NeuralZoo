
from hpelm import ELM
from MetReg.base.base_model import BaseModel
"""
from sklearn_extensions.extreme_learning_machines.elm import (ELMRegressor,
                                                              GenELMRegressor)
from sklearn_extensions.extreme_learning_machines.random_layer import \
    MLPRandomLayer, RandomLayer
"""


class ExtremeLearningRegressor(BaseModel):

    def __init__(self,):
        self.regressor = None

    def fit(self, X, y):
        n_features = X.shape[-1]
        self.regressor = ELM(n_features, 1)
        self.regressor.add_neurons(16, 'sigm')

        self.regressor.train(X, y)
        return self
