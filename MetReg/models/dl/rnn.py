from tensorflow.keras import models
from tensorflow.keras import layers
from MetReg.base.base_model import BaseModel


class RNNRegressor(BaseModel):

    def __init__(self,
                 hidden_layers_sizes=(64,),
                 activation='relu',):
        self.regressor = None
        self.hidden_layers_sizes = hidden_layers_sizes

    def fit(self, X, y):
        n_features = X.shape[-1]
        n_steps = X.shape[-2]

        self.regressor = models.Sequential()
        for i, n_units in enumerate(self.hidden_layers_sizes):
            self.regressor.add(layers.RNN(units=n_units))
        self.regressor.add(layers.Dense(1))

        self.regressor.build(input_shape=(n_steps, n_features))

        self.regressor.compile()
        self.regressor.fit(X, y)
        return self


class lstm:

    def __init__(self): pass

    def __call__(self):

        mdl = models.Sequential()
        mdl.add(tf.keras.layers.LSTM(units=64, input_shape=(10, 3)))
        mdl.add(tf.keras.layers.Dense(1))
        mdl.summary()

        return mdl


class rnn:

    def __init__(self): pass

    def __call__(self):

        mdl = tf.keras.models.Sequential()
        mdl.add(tf.keras.layers.RNN(units=64, input_shape=(10, 3)))
        mdl.add(tf.keras.layers.Dense(1))
        mdl.summary()

        return mdl


class gru:

    def __init__(self): pass

    def __call__(self):

        mdl = tf.keras.models.Sequential()
        mdl.add(tf.keras.layers.GRU(units=64, input_shape=(10, 3)))
        mdl.add(tf.keras.layers.Dense(1))
        mdl.summary()

        return mdl


class bilstm:

    def __init__(self): pass

    def __call__(self):

        mdl = tf.keras.models.Sequential()
        mdl.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=64, input_shape=(10, 3))))
        mdl.add(tf.keras.layers.Dense(1))
        mdl.summary()

        return mdl
