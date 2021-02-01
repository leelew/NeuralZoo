
from hpelm import ELM

"""
from sklearn_extensions.extreme_learning_machines.elm import (ELMRegressor,
                                                              GenELMRegressor)
from sklearn_extensions.extreme_learning_machines.random_layer import \
    MLPRandomLayer, RandomLayer
"""

class elm():

    def __init__(self): 
        self.regressor=None

    def fit(self, X, y):

        self.regressor = ELM(30, 1)
        self.regressor.add_neurons(16, 'sigm')

        self.regressor.train(X, y)
        return self
