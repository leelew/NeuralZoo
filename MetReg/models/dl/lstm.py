import tensorflow as tf


class lstm:

    def __init__(self): pass

    def __call__(self):

        mdl = tf.keras.models.Sequential()
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
