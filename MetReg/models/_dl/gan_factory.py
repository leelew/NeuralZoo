import time
from ast import Str

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
import tqdm
from factory.data.data_loader import DataLoader
from factory.loss import ImageGradientDifferenceLoss, LPLoss
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import Input, Model, Sequential, backend, layers
from tensorflow.keras.layers import (Add, BatchNormalization, Conv2D,
                                     ConvLSTM2D, Dense, Dropout, Flatten,
                                     GlobalAveragePooling2D, Input, Lambda,
                                     MaxPooling2D, ReLU, Reshape, UpSampling2D,
                                     concatenate, multiply)
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras.losses import mean_squared_error

from model.convlstm_factory import SMNet


class GAN():

    """implement of General Advertisal Network constructed of
    ConvLSTM (G) and CNN(D).
    """

    def __init__(self, epochs=50):

        self.epochs = epochs

        # build
        self.gen = self.generator()
        self.disc = self.discriminator()

        self.optimizer = Adam()

    def generator(self):
        inputs = Input(shape=(7, 112, 112, 8))

        # preprocess l4
        outputs, h, c = tf.keras.layers.ConvLSTM2D(16, 5, 
                            padding='same', 
                            kernel_initializer='he_normal', return_state=True)(inputs)
        out = Lambda(lambda x: backend.concatenate([x] * 7, axis=1))(h[:, tf.newaxis])

        out = ConvLSTM2D(16, 3, padding='same', return_sequences=True)(out)
        out = tf.keras.layers.Dense(1)(out)

        mdl = Model(inputs, out)

        mdl.summary()    
        return mdl

    def discriminator(self):

        inputs = Input(shape=(7, 112, 112, 1))
        outputs = tf.squeeze(inputs, axis=-1)
        print(inputs.shape)

        outputs = tf.transpose(outputs, [0, 2, 3, 1])
        print(inputs.shape)
        print('1')
        outputs = Conv2D(16, 3, 2)(outputs)
        print(outputs.shape)

        outputs = Conv2D(32, 3, 2)(outputs)
        print(outputs.shape)

        outputs = Conv2D(16, 3, 2)(outputs)
        print(outputs.shape)

        outputs = Conv2D(1, 1, 2)(outputs)
        print(outputs.shape)

        outputs = Flatten()(outputs)
        print(outputs.shape)
        outputs = Dense(1, activation='sigmoid')(outputs)

        mdl = Model(inputs, outputs)
        mdl.summary()

        return mdl

    def G_loss(self, y_truth, G_pred):
        """generator loss construct of square-error loss
        """
        G_mse_loss = MeanSquaredError()(y_truth, G_pred)
        # print(tf.print(G_mse_loss))
        # print(tf.print(G_bce_loss))
        return G_mse_loss

    def D_loss(self, D_truth, D_pred):
        """discriminator loss
        Args:
            D_truth ([type]): [description]
            D_pred ([type]): [description]
        """
        D_truth_loss = BinaryCrossentropy()(tf.ones_like(D_truth), D_truth)
        D_pred_loss = BinaryCrossentropy()(tf.zeros_like(D_pred), D_pred)
        return D_truth_loss, D_pred_loss

    @tf.function
    def train_step(self,
                   X,
                   y,
                   generator,
                   discriminator,
                   optimizer):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            # forward
            G_pred = generator(X, training=True)
            D_truth = discriminator(y, training=True)
            D_pred = discriminator(G_pred, training=True)

            # loss
            D_truth_loss_, D_pred_loss_ = self.D_loss(D_truth, D_pred)
            D_loss_ = D_truth_loss_ + D_pred_loss_
            G_loss_ = self.G_loss(y, G_pred)

            tf.print(D_pred_loss_)
            tf.print(G_loss_)
            G_loss_ += 0.05*D_pred_loss_

            # gradient
            D_gradient = disc_tape.gradient(D_loss_, discriminator.trainable_variables)
            G_gradient = gen_tape.gradient(G_loss_, generator.trainable_variables)

            # gradient descent
            optimizer.apply_gradients(zip(D_gradient, discriminator.trainable_variables))
            optimizer.apply_gradients(zip(G_gradient, generator.trainable_variables))


    def fit(self, X, y, epochs, batch_size):
        #TODO: change X,y to tf.dataset
        #dataset = DataLoader(epochs, batch_size).fit(X, y)
        # construct `tf.data.Dataset`
        dataset = tf.data.Dataset.from_tensor_slices(
            (X, y)).shuffle(5000).batch(batch_size, drop_remainder=True)

        for epoch in range(self.epochs):
            print(epoch)
            for i, (X_batch, y_batch) in enumerate(dataset):
                #time1 = time.time()
                self.train_step(X_batch, y_batch, self.gen, self.disc, self.optimizer)
                #time2 = time.time()
                #predict = self.gen(X_batch)
                #print(LPLoss()(y, predict))

            
            # Plot the progress
            #print ("%d %d [G MSE loss: %f] cost %d" % (epoch, i, MeanSquaredError(y_batch, self.#gen(X_batch)), time2-time1))

    def train(self, x_train, y_train):

        S, T, H, W, F = x_train.shape

        train_ds = np.concatenate(
            [x_train.reshape(S, H, W, T*F), y_train[:, :, :, np.newaxis]], axis=-1)
        np.random.shuffle(train_ds)

        index = np.array(train_ds[:, :, :, :-1]).reshape(S, T, H, W, F)
        label = np.array(train_ds[:, :, :, -1]).reshape(S, H, W, 1)
        print(index.shape)

        # construct `tf.data.Dataset`
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (index, label)).shuffle(100000).batch(
                    32, drop_remainder=True)

        for epoch in tqdm(range(self.epochs)):
            for x, y in train_dataset:

                self.train_step(self.gen, self.disc, self.optimizer,
                                x, y)

    def predict(self, x_test):
        return self.gen(x_test)

#TODO:Add compile part into models in model.factory
class GAN1():
    def __init__(self) -> None:

        #self.gen_optimizer = tf.keras.optimizers.Adam(1e-4)
        #self.disc_optimizer = tf.keras.optimizers.Adam(1e-4)
        # build the generator and discriminator
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        # ----------------------------------
        # construct discriminator
        self.generator.trainable = False
        self.discriminator.trainable = True

        # generator inputs 
        in_l3 = Input(shape=(7, 112, 112, 1))
        in_l4 = Input(shape=(7, 112, 112, 8))
        fake_img = self.generator([in_l3, in_l4])

        # 
        real_img = Input(shape=(7, 112, 112, 1))
        fake = self.discriminator(fake_img)
        valid = self.discriminator(real_img)

        self.disc_model = Model(inputs=[in_l3, in_l4, real_img], outputs=[fake, valid])
        #FIXME: compile discriminator 
        self.disc_model.compile(loss=['binary_crossentropy', 'binary_crossentropy'], optimizer=Adam(1e-4))
        
        # ---------------------------------
        self.discriminator.trainable = False
        self.generator.trainable = True


        # 
        in_l3 = Input(shape=(7, 112, 112, 1))
        in_l4 = Input(shape=(7, 112, 112, 8))
        fake_img = self.generator([in_l3, in_l4])
        valid = self.discriminator(fake_img)
        self.gan = Model([in_l3, in_l4], valid)
        self.gan.compile(loss='binary_crossentropy', optimizer=Adam())

    

    def generator_loss(self, y_true, y_pred):
        return LPLoss()(y_true, y_pred) #+ ImageGradientDifferenceLoss()(y_true, y_pred)

    #TODO: add to loss factory
    def loss_hinge_disc(self, score_generated, score_real):
        """Discriminator hinge loss."""
        l1 = K.relu(1.0 - score_real)
        loss = K.mean(l1)
        l2 = K.relu(1.0 + score_generated)
        loss += K.mean(l2)
        return loss

    def fit(self, X, y, epochs, batch_size):
        #TODO: change X,y to tf.dataset
        dataset = DataLoader(epochs, batch_size).fit(X, y)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        
        for epoch in range(epochs):
            for i, (X_batch, y_batch) in enumerate(dataset):
                time1 = time.time()

                for _ in range(2):
                    d_loss = self.disc_model.train_on_batch([X_batch, y_batch], [fake, valid])
            
                g_loss = self.gan.train_on_batch(X_batch, valid)

                time2 = time.time()

            predict = self.generator(X)

            # Plot the progress
            print ("%d %d [G MSE loss: %f] cost %d" % (epoch, i, LPLoss()(y, self.generator(X)), time2-time1))

    def generate_images(model, test_input):
        prediction = model(test_input)

        plt.figure(figsize=(12, 12))

        display_list = [test_input[0], prediction[0]]
        title = ['Input Image', 'Predicted Image']

        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()

    def predict(self, X):
        return self.generator(X)

    def build_generator(self): return SMNet()

    def build_discriminator(self):
        inputs = Input(shape=(7, 112, 112, 1))
        #out = Conv2D(256, kernel_size=3, strides=2, padding="same")(inputs)
        #out = LeakyReLU(alpha=0.2)(out)
        #out = Conv2D(128, kernel_size=3, strides=2, padding='same')(out)
        #out = LeakyReLU(alpha=0.2)(out)
        #out = tf.keras.layers.Permute((4, 2, 3, 1))(inputs)
        #out = tf.squeeze(out, axis=1)
        #out = Conv2D(8, kernel_size=3, strides=2, padding='same')(out)
        #out = LeakyReLU(alpha=0.2)(out)
        #out = Conv2D(32, kernel_size=3, strides=2, padding='same')(out)
        #out = LeakyReLU(alpha=0.2)(out)
        out = ConvLSTM2D(4, 3, padding='same', strides=2, return_sequences=True)(inputs)
        out = ConvLSTM2D(2, 3, padding='same', strides=2, return_sequences=True)(out)
        out = ConvLSTM2D(1, 3, padding='same', strides=2, return_sequences=True)(out)
        out = Flatten()(out)

        #out = Dense(1024)(out)
        out = Dense(1)(out)
        mdl = Model(inputs, out)
        mdl.summary()
        return mdl



#TODO:Add compile part into models in model.factory
class GAN0():
    def __init__(self) -> None:

        self.gen_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.disc_optimizer = tf.keras.optimizers.Adam(1e-4)
        # build the generator and discriminator
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        """
        # ----------------------------------
        # construct discriminator
        self.generator.trainable = False
        self.discriminator.trainable = True

        # generator inputs 
        in_l3 = Input(shape=(7, 112, 112, 1))
        in_l4 = Input(shape=(7, 112, 112, 8))
        fake_img = self.generator([in_l3, in_l4])

        # 
        real_img = Input(shape=(7, 112, 112))
        fake = self.discriminator(fake_img)
        valid = self.discriminator(real_img)

        #FIXME: compile discriminator 
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam())
        
        # ---------------------------------
        self.discriminator.trainable = False
        self.generator.trainable = True


        # 
        in_l3 = Input(shape=(7, 112, 112, 1))
        in_l4 = Input(shape=(7, 112, 112, 8))
        fake_img = self.generator([in_l3, in_l4])
        valid = self.discriminator(fake_img)
        self.gan = Model([in_l3, in_l4], valid)
        self.gan.compile(loss='binary_crossentropy', optimizer=Adam())
        """

    def generator_loss(self, y_true, y_pred):
        return LPLoss()(y_true, y_pred) #+ ImageGradientDifferenceLoss()(y_true, y_pred)

    #TODO: add to loss factory
    def loss_hinge_disc(self, score_generated, score_real):
        """Discriminator hinge loss."""
        l1 = K.relu(1.0 - score_real)
        loss = K.mean(l1)
        l2 = K.relu(1.0 + score_generated)
        loss += K.mean(l2)
        return loss

    @tf.function
    def train_step(self, X, y):
        self.global_iteration = 0
        self.n_iter_discriminator = 2
        # persistent is set to True because the tape is used 
        # more than once to calculate the gradients.
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            
            # #######################
            # optimize discriminator
            # #######################
            predictions = self.generator(X, training=True) # predictions (s, t2, lat, lon, 1)
            # X: (s, t1, lat, lon, 1)
            # y: (s, t2, lat, lon, 1)
            # cat along time dimension
            generated_frame = predictions#tf.concat([X, predictions], axis=1)
            real_frame = tf.cast(y, tf.float32)#tf.concat([X, y], axis=1)

            concatenate_inputs = tf.concat([real_frame, generated_frame], axis=0)            
            concatenate_outputs = self.discriminator(concatenate_inputs, training=True)

            score_real, score_generated = tf.split(concatenate_outputs, 2, axis=0)
            discriminator_loss = self.loss_hinge_disc(score_real, score_generated)+0.0
            generator_loss = self.generator_loss(real_frame+0.0, generated_frame+0.0)
            #generator_loss += 0.05*K.mean(score_generated)+0.0

            gen_gradients = gen_tape.gradient(generator_loss, self.generator.trainable_variables)
            disc_gradients = disc_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
            
            self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
            self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

    def fit(self, X, y, epochs, batch_size):
        #TODO: change X,y to tf.dataset
        dataset = DataLoader(epochs, batch_size).fit(X, y)

        for epoch in range(epochs):
            for i, (X_batch, y_batch) in enumerate(dataset):
                time1 = time.time()
                self.train_step(X_batch, y_batch+0.0)
                time2 = time.time()

                predict = self.generator(X)
                plt.figure(figsize=(6, 12))
                plt.subplot(1, 2, 1)
                plt.imshow(predict[0, 0, :,:,0])
                plt.subplot(1, 2, 2)
                plt.imshow(y[0, 0, :,:,0])
                plt.axis('off')
                plt.savefig('step_{}.pdf'.format(i))

                # Plot the progress
                print ("%d %d [G MSE loss: %f] cost %d" % (epoch, i, LPLoss()(y, self.generator(X)), time2-time1))

    def generate_images(model, test_input):
        prediction = model(test_input)

        plt.figure(figsize=(12, 12))

        display_list = [test_input[0], prediction[0]]
        title = ['Input Image', 'Predicted Image']

        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()

    def predict(self, X):
        return self.generator(X)

    def build_generator(self): return SMNet()

    def build_discriminator(self):
        inputs = Input(shape=(7, 112, 112, 1))
        #out = Conv2D(256, kernel_size=3, strides=2, padding="same")(inputs)
        #out = LeakyReLU(alpha=0.2)(out)
        #out = Conv2D(128, kernel_size=3, strides=2, padding='same')(out)
        #out = LeakyReLU(alpha=0.2)(out)
        #out = tf.keras.layers.Permute((4, 2, 3, 1))(inputs)
        #out = tf.squeeze(out, axis=1)
        #out = Conv2D(8, kernel_size=3, strides=2, padding='same')(out)
        #out = LeakyReLU(alpha=0.2)(out)
        #out = Conv2D(32, kernel_size=3, strides=2, padding='same')(out)
        #out = LeakyReLU(alpha=0.2)(out)
        out = ConvLSTM2D(4, 3, padding='same', strides=2, return_sequences=True)(inputs)
        out = ConvLSTM2D(2, 3, padding='same', strides=2, return_sequences=True)(out)
        out = ConvLSTM2D(1, 3, padding='same', strides=2, return_sequences=True)(out)
        out = Flatten()(out)

        #out = Dense(1024)(out)
        out = Dense(1)(out)
        mdl = Model(inputs, out)
        mdl.summary()
        return mdl
