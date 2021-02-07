# -----------------------------------------------------------------------------
#              General Advertisal Network Model Repository (GANMR)            #
# -----------------------------------------------------------------------------
# author: Lu Li                                                               #
# mail: lilu35@mail2.sysu.edu.cn                                              #
# -----------------------------------------------------------------------------
# This Repo                                                                   #
# -----------------------------------------------------------------------------

from __future__ import print_function

import abc
import warnings

import numpy as np
import tensorflow as tf
from metrics import Metrics
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (BatchNormalization, Conv2D, ConvLSTM2D,
                                     Dense, Flatten)
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

np.random.seed(1)
tf.compat.v1.set_random_seed(13)


class GAN_ConvLSTM():
    """implement of General Advertisal Network.

    This class creates a general advertisal network that using ConvLSTM as
    generator and simple NN as discriminator.
    """

    def __init__(self, epochs=20):
        if isinstance(epochs, float):
            epochs = int(epochs)
        self.epochs = epochs

        # build
        self.gen = self.generator()
        self.disc = self.discriminator()

        self.optimizer = Adam()

    def generator(self):

        mdl = Sequential()
        mdl.add(
            ConvLSTM2D(filters=16,
                       kernel_size=(3, 3),
                       padding='same',
                       activation='relu',
                       input_shape=(10, 8, 8, 3)))
        # mdl.add(BatchNormalization())

        mdl.add(Dense(units=1))
        mdl.summary()

        return mdl

    def discriminator(self):

        mdl = Sequential()

        mdl.add(Flatten())
        mdl.add(Dense(units=1, activation='sigmoid'))

        """
        mdl.add(
            Conv2D(filters=16,
                   kernel_size=(3, 3),
                   padding='same',
                   input_shape=(8, 8, 1)))
        mdl.add(Flatten())
        mdl.add(Dense(units=1))
        """
        mdl.build((None, 8, 8, 1))
        mdl.summary()

        return mdl

    @staticmethod
    def G_loss(y_truth, G_pred, D_pred):
        """generator loss construct of square-error loss
        """
        G_mse_loss = MeanSquaredError()(y_truth, G_pred)
        G_bce_loss = BinaryCrossentropy()(tf.zeros_like(D_pred), D_pred)
        # print(tf.print(G_mse_loss))
        # print(tf.print(G_bce_loss))
        return G_mse_loss - 0.0003*G_bce_loss

    @staticmethod
    def D_loss(D_truth, D_pred):
        """discriminator loss

        Args:
            D_truth ([type]): [description]
            D_pred ([type]): [description]
        """
        D_truth_loss = BinaryCrossentropy()(tf.ones_like(D_truth), D_truth)
        D_pred_loss = BinaryCrossentropy()(tf.zeros_like(D_pred), D_pred)
        return D_truth_loss + D_pred_loss

    @tf.function
    def train_step(self, gen, disc, optimizer, x_train, y_train):
        """[summary]

        Args:
            x_train ([type]): [description]
            y_train ([type]): [description]
        """
        with tf.GradientTape() as gen_tape, \
                tf.GradientTape() as disc_tape:

            # forward
            G_pred = gen(x_train, training=True)
            D_truth = disc(y_train, training=True)
            D_pred = disc(G_pred, training=True)

            # loss
            D_loss = GAN_ConvLSTM.D_loss(D_truth, D_pred)
            G_loss = GAN_ConvLSTM.G_loss(y_train, G_pred, D_pred)
            #print(tf.print(G_loss), tf.print(D_loss))

            # gradient
            D_gradient = disc_tape.gradient(D_loss, disc.trainable_variables)
            G_gradient = gen_tape.gradient(G_loss, gen.trainable_variables)

            # gradient descent
            optimizer.apply_gradients(zip(D_gradient,
                                          disc.trainable_variables))
            optimizer.apply_gradients(zip(G_gradient, gen.trainable_variables))

    def train(self, x_train, y_train):

        S, T, H, W, F = x_train.shape

        train_ds = np.concatenate(
            [x_train.reshape(S, H, W, T * F), y_train[:, :, :, np.newaxis]],
            axis=-1)
        np.random.shuffle(train_ds)

        index = np.array(train_ds[:, :, :, :-1]).reshape(S, T, H, W, F)
        label = np.array(train_ds[:, :, :, -1]).reshape(S, H, W, 1)

        # construct `tf.data.Dataset`
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (index, label)).shuffle(100000).batch(32, drop_remainder=True)

        for epoch in tqdm(range(self.epochs)):
            for x, y in train_dataset:

                self.train_step(self.gen, self.disc, self.optimizer, x, y)

    def evaluate(self, x_valid, y_valid):

        y_predict = self.gen.predict(x_valid)
        Metrics(y_valid, y_predict).get_sklearn_metrics(y_valid, y_predict)


if __name__ == "__main__":

    GAN_ConvLSTM()
