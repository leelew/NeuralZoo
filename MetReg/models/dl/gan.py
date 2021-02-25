import numpy as np
import tensorflow as tf
import tqdm
from MetReg.train.loss import D_loss, G_loss
from tensorflow.keras import Model, Sequential, activations, layers
from MetReg.data.data_loader import Data_loader
from tensorflow.keras.optimizers import Adam

tf.compat.v1.set_random_seed(1)


class GANConvLSTMRegressor(Model):

    def __init__(self):
        super().__init__()

        self.generator = Sequential(layers=[
            layers.ConvLSTM2D(filters=16,
                              kernel_size=(
                                  3, 3),
                              padding='same',
                              activation='relu'),
            layers.Dense(1)
        ])

        self.discriminator = Sequential(
            layers=[
                layers.Flatten(),
                layers.Dense(1)
            ])

    def train_step(self,
                   X,
                   y,
                   generator,
                   discriminator,
                   optimizer):
        with tf.GradientTape() as gen_tape, \
                tf.GradientTape() as disc_tape:

            # forward
            G_pred = generator(X, training=True)
            D_truth = discriminator(y, training=True)
            D_pred = discriminator(G_pred, training=True)

            # loss
            D_loss_ = D_loss(D_truth, D_pred)
            G_loss_ = G_loss(y, G_pred, D_pred)

            # gradient
            D_gradient = discriminator_tape.gradient(
                D_loss_, discriminator.trainable_variables)
            G_gradient = generator_tape.gradient(
                G_loss_, generator.trainable_variables)

            # gradient descent
            optimizer.apply_gradients(zip(D_gradient,
                                          discriminator.trainable_variables))
            optimizer.apply_gradients(zip(G_gradient,
                                          generator.trainable_variables))

    def fit(self,
            X,
            y,
            epochs,
            batch_size, optimizer=Adam()):
        train_ds = Data_loader(X, y, epochs=epochs,
                               batch_size=batch_size).get_tf_dataset()

        for epoch in tqdm(range(epochs)):
            for x, y in train_ds:
                train_step(self.generator, self.discriminator, optimizer, x, y)

        return self
