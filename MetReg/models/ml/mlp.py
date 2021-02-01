from MetReg.base.base_model import base_model
from sklearn.neural_network import MLPRegressor


class mlp(base_model):

    def __init__(self, 
                hidden_layers_sizes=(100,),
                activation='relu',):

        self.hidden_layers_sizes = hidden_layers_sizes
        self.activation = activation
        self.regressor = None

    def fit(self, X, y):

        self.regressor = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers_sizes,
            activation=self.activation,
        )

        self.regressor.fit(X, y)
        return self
