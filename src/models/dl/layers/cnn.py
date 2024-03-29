import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.models import Sequential


def VGG():
    """
    VGG created by Visual Geometry Group,
    raised a method to create Deep model by repeat add vgg block
    could read from https://arxiv.org/abs/1409.1556
    """

    conv_map = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

    vgg = tf.keras.models.Sequential()
    for i, j in conv_map:
        vgg.add(self.vgg_block(i, j))
    vgg.add(tf.keras.layers.Flatten())
    vgg.add(tf.keras.layers.Dense(4096, activation='relu'))
    vgg.add(tf.keras.layers.Dropout(0.5))
    vgg.add(tf.keras.layers.Dense(4096, activation='relu'))
    vgg.add(tf.keras.layers.Dropout(0.5))
    vgg.add(tf.keras.layers.Dense(10, activation='sigmoid'))

    vgg.build((None, 128, 128, 1))

    return vgg

    def vgg_block(self, num_convs, num_channels):
        model = tf.keras.models.Sequential()
        for _ in range(num_convs):
            model.add(
                tf.keras.layers.Conv2D(num_channels,
                                       kernel_size=3,
                                       padding='same',
                                       activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        return model

class BaseCNNRegressor(Model):
    def __init__(self):
        super().__init__()
        self.cnn3 = layers.Conv2D(filters=128,
                                  kernel_size=(3, 3),
                                  activation='relu',
                                  padding='same')

        self.cnn4 = layers.Conv2D(filters=64,
                                  kernel_size=(3, 3),
                                  activation='relu',
                                  padding='same')

        self.cnn5 = layers.Conv2D(filters=32,
                                  kernel_size=(3, 3),
                                  activation='relu',
                                  padding='same')

        self.dense5 = layers.Dense(1)

    def call(self, inputs):

        x = self.cnn3(inputs)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.dense5(x)

        return x


class LeNet(Model):
    """default CNN, created by Yann LeCun.
        https://www.mitpressjournals.org/doi/abs/10.1162/neco.1989.1.4.541
    """
    def __init__(self):
        self.cnn1 = layers.Conv2D(
            filters=6,
            kernel_size=5,
            activation='sigmoid',
        )
        self.avgpool1 = layers.AvgPool2D(pool_size=2, strides=2)

        self.cnn2 = layers.Conv2D(filters=16,
                                  kernel_size=5,
                                  activation='sigmoid')
        self.avgpool2 = layers.AvgPool2D(pool_size=2, strides=2)
        self.flat = layers.Flatten()
        self.dense1 = layers.Dense(120, activation='sigmoid')
        self.dense2 = layers.Dense(84, activation='sigmoid')
        self.dense3 = layers.Dense(10, activation='sigmoid')

    def call(self, inputs):
        x = self.cnn1(inputs)
        x = self.avgpool1(x)
        x = self.cnn2(x)
        x = self.avgpool2(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


class vanilla():
    """implement of vanilla Convolutional Neural Network.

    Args:

    Returns:

    """
    def __init__(self):
        pass

    def AlexNet(self):
        """
        same with LeNet, but introduce ReLU, dropout method etc. 
        created by Alex Krizhevsky. http://papers.nips.cc/paper/
        4824-imagenet-classification-with-deep-convolutional-neural-network
        """

        model = tf.keras.models.Sequential()
        # if you want to test MNIST, you must changed picture to (224,224).
        model.add(
            tf.keras.layers.Conv2D(filters=96,
                                   kernel_size=11,
                                   strides=4,
                                   activation='relu',
                                   input_shape=[128, 128, 1]))
        model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
        model.add(
            tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=5,
                                   padding='same',
                                   activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
        model.add(
            tf.keras.layers.Conv2D(filters=384,
                                   kernel_size=5,
                                   padding='same',
                                   activation='relu'))
        model.add(
            tf.keras.layers.Conv2D(filters=384,
                                   kernel_size=5,
                                   padding='same',
                                   activation='relu'))
        model.add(
            tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=5,
                                   padding='same',
                                   activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(4096, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(4096, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(10, activation='sigmoid'))

        return model

    def VGG(self):
        """
        VGG created by Visual Geometry Group,
        raised a method to create Deep model by repeat add vgg block
        could read from https://arxiv.org/abs/1409.1556
        """

        conv_map = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

        vgg = tf.keras.models.Sequential()
        for i, j in conv_map:
            vgg.add(self.vgg_block(i, j))
        vgg.add(tf.keras.layers.Flatten())
        vgg.add(tf.keras.layers.Dense(4096, activation='relu'))
        vgg.add(tf.keras.layers.Dropout(0.5))
        vgg.add(tf.keras.layers.Dense(4096, activation='relu'))
        vgg.add(tf.keras.layers.Dropout(0.5))
        vgg.add(tf.keras.layers.Dense(10, activation='sigmoid'))

        vgg.build((None, 128, 128, 1))

        return vgg

    def vgg_block(self, num_convs, num_channels):
        model = tf.keras.models.Sequential()
        for _ in range(num_convs):
            model.add(
                tf.keras.layers.Conv2D(num_channels,
                                       kernel_size=3,
                                       padding='same',
                                       activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        return model

    def NIN(self):
        """Network in network remove full connected layers, which easy to overfit,
        and changed as Nin block and global average pooling
        could read from https://arxiv.org/abs/1312.4400
        """
        nin = tf.keras.models.Sequential()
        nin.add(self.nin_block(96, kernel_size=11, strides=4, padding='valid'))
        nin.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
        nin.add(self.nin_block(256, kernel_size=5, strides=1, padding='same'))
        nin.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
        nin.add(self.nin_block(384, kernel_size=3, strides=1, padding='same'))
        nin.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
        nin.add(tf.keras.layers.Dropout(0.5))
        nin.add(self.nin_block(10, kernel_size=3, strides=1, padding='same'))
        nin.add(tf.keras.layers.GlobalAveragePooling2D())
        nin.add(tf.keras.layers.Flatten())

        nin.build((None, 128, 128, 1))

        return nin

    def nin_block(self, num_channels, kernel_size, strides, padding):

        nin = tf.keras.models.Sequential()
        nin.add(
            tf.keras.layers.Conv2D(num_channels,
                                   kernel_size,
                                   strides=strides,
                                   padding=padding,
                                   activation='relu'))
        nin.add(
            tf.keras.layers.Conv2D(num_channels,
                                   kernel_size=1,
                                   activation='relu'))
        nin.add(
            tf.keras.layers.Conv2D(num_channels,
                                   kernel_size=1,
                                   activation='relu'))

        return nin

    def GoogLeNet(self):
        """
        inception model created by google
        https://arxiv.org/pdf/1409.4842.pdf
        """

        model01 = tf.keras.models.Sequential()
        model01.add(
            tf.keras.layers.Conv2D(64,
                                   kernel_size=7,
                                   strides=2,
                                   padding='same',
                                   activation='relu'))
        model01.add(
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))

        model02 = tf.keras.models.Sequential()
        model02.add(
            tf.keras.layers.Conv2D(64,
                                   kernel_size=1,
                                   padding='same',
                                   activation='relu'))
        model02.add(
            tf.keras.layers.Conv2D(192,
                                   kernel_size=3,
                                   padding='same',
                                   activation='relu'))
        model02.add(
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))

        model03 = tf.keras.models.Sequential()
        model03.add(inception(64, (96, 128), (16, 32), 32))
        model03.add(inception(128, (128, 192), (32, 96), 64))
        model03.add(
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))

        model04 = tf.keras.models.Sequential()
        model04.add(inception(192, (96, 208), (16, 48), 64))
        model04.add(inception(160, (112, 224), (24, 64), 64))
        model04.add(inception(128, (128, 256), (24, 64), 64))
        model04.add(inception(112, (144, 288), (32, 64), 64))
        model04.add(inception(256, (160, 320), (32, 128), 128))
        model04.add(
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))

        model05 = tf.keras.models.Sequential()
        model05.add(inception(256, (160, 320), (32, 128), 128))
        model05.add(inception(384, (192, 384), (48, 128), 128))
        model05.add(tf.keras.layers.GlobalAvgPool2D())

        googlenet = tf.keras.models.Sequential([
            model01, model02, model03, model04, model05,
            tf.keras.layers.Dense(10)
        ])
        googlenet.build([None, 128, 128, 1])

        return googlenet

    def MobileNet(self):
        """https://arxiv.org/abs/1704.04861
        """
        # not complete mobilenet, only deepwise seperable convolutional.
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Conv2D(filters=6,
                                   kernel_size=5,
                                   activation='sigmoid',
                                   input_shape=[28, 28, 1]))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        model.add(
            tf.keras.layers.SeparableConv2D(filters=16,
                                            kernel_size=5,
                                            activation='sigmoid'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(120, activation='sigmoid'))
        model.add(tf.keras.layers.Dense(84, activation='sigmoid'))
        model.add(tf.keras.layers.Dense(10, activation='sigmoid'))

        return model

    def TCN(self):
        pass


class inception(tf.keras.layers.Layer):
    def __init__(self, c1, c2, c3, c4):
        super().__init__()

        self.p1_1 = tf.keras.layers.Conv2D(c1,
                                           kernel_size=1,
                                           activation='relu',
                                           padding='same')

        self.p2_1 = tf.keras.layers.Conv2D(c2[0],
                                           kernel_size=1,
                                           padding='same',
                                           activation='relu')
        self.p2_2 = tf.keras.layers.Conv2D(c2[1],
                                           kernel_size=3,
                                           padding='same',
                                           activation='relu')

        self.p3_1 = tf.keras.layers.Conv2D(c3[0],
                                           kernel_size=1,
                                           padding='same',
                                           activation='relu')
        self.p3_2 = tf.keras.layers.Conv2D(c3[1],
                                           kernel_size=5,
                                           padding='same',
                                           activation='relu')

        self.p4_1 = tf.keras.layers.MaxPool2D(pool_size=3,
                                              padding='same',
                                              strides=1)
        self.p4_2 = tf.keras.layers.Conv2D(c4,
                                           kernel_size=1,
                                           padding='same',
                                           activation='relu')

    def call(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return tf.concat([p1, p2, p3, p4], axis=-1)  # 在通道维上连结输出


class resid(tf.keras.Model):
    def __init__(self, num_channels, pointwise=False, strides=1, **kwargs):
        super(resid, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(num_channels,
                                            padding='same',
                                            kernel_size=3,
                                            strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(num_channels,
                                            kernel_size=3,
                                            padding='same')

        self.bn2 = tf.keras.layers.BatchNormalization()

        if pointwise:
            self.conv3 = tf.keras.layers.Conv2D(num_channels,
                                                kernel_size=1,
                                                strides=strides)

        else:
            self.conv3 = None

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return tf.keras.activations.relu(Y + X)


if __name__ == "__main__":

    model = Convolutional_Neuron_Network_Regression()
    AlexNet = model.AlexNet()
    AlexNet.summary()
