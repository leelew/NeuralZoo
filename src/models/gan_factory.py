import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from factory.layers import DIConvLSTM2D
from tensorflow.keras import Input, Model, backend, layers
from tensorflow.keras.layers import (Add, BatchNormalization, Conv2D,
                                     ConvLSTM2D, Dense, Dropout, Flatten,
                                     GlobalAveragePooling2D, Input, Lambda,
                                     MaxPooling2D, ReLU, Reshape,
                                     UpSampling2D, concatenate, multiply)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU

from model.convlstm_factory import SMNet
from factory.loss import ImageGradientDifferenceLoss, LPLoss
from factory.data.data_loader import DataLoader
import time

#TODO:Add compile part into models in model.factory
class GAN():
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
        return LPLoss()(y_true, y_pred) + ImageGradientDifferenceLoss()(y_true, y_pred)

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
            print(score_generated.shape)
            generator_loss = self.generator_loss(real_frame+0.0, generated_frame+0.0)
            generator_loss += 0.05*K.mean(score_generated)+0.0

            print(generator_loss)
            print(discriminator_loss)
            gen_gradients = gen_tape.gradient(generator_loss, self.generator.trainable_variables)
            disc_gradients = disc_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
            
            self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
            self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

    def fit(self, X, y, epochs, batch_size):
        #TODO: change X,y to tf.dataset
        dataset = DataLoader(epochs, batch_size).fit(X, y)

        for epoch in range(epochs):
            for i, (X_batch, y_batch) in enumerate(dataset):
                #time1 = time.time()
                self.train_step(X_batch, y_batch+0.0)
                #time2 = time.time()
                #print('{} step, cost {}'.format(i, time2-time1))
            # Plot the progress
            print ("%d [G MSE loss: %f]" % (epoch, LPLoss()(y, self.generator(X))))

    def predict(self, X):
        return self.generator(X)

    def build_generator(self): return SMNet()

    def build_discriminator(self):
        inputs = Input(shape=(7, 112, 112, 1))
        #out = Conv2D(256, kernel_size=3, strides=2, padding="same")(inputs)
        #out = LeakyReLU(alpha=0.2)(out)
        #out = Conv2D(128, kernel_size=3, strides=2, padding='same')(out)
        #out = LeakyReLU(alpha=0.2)(out)
        out = tf.keras.layers.Permute((4, 2, 3, 1))(inputs)
        out = tf.squeeze(out, axis=1)
        print(out.shape)
        out = Conv2D(64, kernel_size=3, strides=2, padding='same')(out)
        out = LeakyReLU(alpha=0.2)(out)
        out = Conv2D(32, kernel_size=3, strides=2, padding='same')(out)
        out = LeakyReLU(alpha=0.2)(out)
        print(out.shape)
        out = Flatten()(out)
        print(out.shape)

        #out = Dense(1024)(out)
        out = Dense(1)(out)
        mdl = Model(inputs, out)
        mdl.summary()
        return mdl
