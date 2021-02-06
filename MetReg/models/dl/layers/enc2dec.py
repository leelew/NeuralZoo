from tensorflow.keras import layers, Model



class Encoder(Model):

    def __init__(self):
        super().__init__()

        self.dense1 = layers.Dense(128, activation='tanh')
        self.dense2 = layers.Dense(64, activation='tanh')
        self.dense3 = layers.Dense(32, activation='tanh')
        self.dense4 = layers.Dense(2, activation='sigmoid')

    def call(self, X, training=False):
        out = self.dense1(X)
        out = self.dense2(out)
        out = self.dense3(out)
        out = self.dense4(out)
        return out

class Decoder(Model):

    def __init__(self):
        super().__init__()

        self.dense1 = layers.Dense(16, activation='tanh')
        self.dense2 = layers.Dense(64, activation='tanh')
        self.dense3 = layers.Dense(128, activation='tanh')
        self.dense4 = layers.Dense(784, activation='sigmoid')

    def call(self, X, training=False):
        out = self.dense1(X)
        out = self.dense2(out)
        out = self.dense3(out)
        out = self.dense4(out)
        return out


class Autoencoder(Model):

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, x, training=False):
        out = self.encoder(x, training)
        out = self.decoder(out, training)
        return out


class ED_ConvLSTM(tf.keras.layers.Layer):
    """encoder-decoder ConvLSTM"""

    def __init__(self, output_len):
        super().__init__()

        self.output_len = output_len

        self.convlstm_1 = tf.keras.layers.ConvLSTM2D(
            32, (3, 3), activation='tanh',
            padding='same', return_sequences=True)
        self.bn_1 = tf.keras.layers.BatchNormalization()

        self.convlstm_2 = tf.keras.layers.ConvLSTM2D(
            32, (3, 3), activation='tanh',
            padding='same', return_sequences=False)
        self.bn_2 = tf.keras.layers.BatchNormalization()

        self.convlstm_3 = tf.keras.layers.ConvLSTM2D(
            32, (3, 3), activation='tanh',
            padding='same', return_sequences=True)
        self.bn_3 = tf.keras.layers.BatchNormalization()

        self.convlstm_4 = tf.keras.layers.ConvLSTM2D(
            1, (3, 3), activation='relu',
            padding='same', return_sequences=True)

    def call(self, inputs):
        """Exec

        .. rubic:: process loop
                   0. convlstm & bn (encoder)
                   1. transform last state to decoder
                   2. decoder
        """
        x = self.convlstm_1(inputs)
        x = self.bn_1(x)

        x = self.convlstm_2(x)
        x = self.bn_2(x)

        x = tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.concatenate(
                [x[:, np.newaxis, :, :, :]]*self.output_len, axis=1))(x)

        x = self.convlstm_3(x)
        x = self.bn_3(x)

        x = self.convlstm_4(x)

        return x