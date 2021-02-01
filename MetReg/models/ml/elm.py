
from hpelm import ELM
from sklearn_extensions.extreme_learning_machines.elm import (ELMRegressor,
                                                              GenELMRegressor)
from sklearn_extensions.extreme_learning_machines.random_layer import \
    MLPRandomLayer, RandomLayer


class elm:

    def __init__(self): pass

    def __call__(self):
        # mdl = GenELMRegressor(hidden_layer=MLPRandomLayer(
        #    n_hidden=128, activation_func='sigmoid'))
        mdl = ELM(30, 1)
        mdl.add_neurons(16, 'sigm')
        #mdl.add_neurons(64, 'sigm')
        #mdl.add_neurons(32, 'sigm')
        return mdl
