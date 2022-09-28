import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Input,
                                     MaxPooling2D, UpSampling2D, concatenate)
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.convolutional import Conv


def unet5(
          filter_size=3,
          n_filters_factor=1,
          n_forecast_months=7,
          n_output_classes=1):

    inputs = Input(shape=(112, 112, 8))

    conv1 = Conv2D(np.int(64 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(np.int(64 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv1)
    bn1 = BatchNormalization(axis=-1)(conv1)  # 112, 112, 64
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)  # 56, 56, 64

    conv2 = Conv2D(np.int(128 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(np.int(128 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv2)
    bn2 = BatchNormalization(axis=-1)(conv2)  # 56, 56, 128
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)  # 28, 28, 128

    conv3 = Conv2D(np.int(256 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(np.int(256 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv3)
    bn3 = BatchNormalization(axis=-1)(conv3)  # 28, 28, 256

    up4 = Conv2D(np.int(128 * n_filters_factor),
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(
                     size=(2, 2), interpolation='nearest')(bn3))  # 56, 56 128
    merge4 = concatenate([bn2, up4], axis=3)
    conv4 = Conv2D(np.int(128 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge4)
    conv4 = Conv2D(np.int(128 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv4)
    bn4 = BatchNormalization(axis=-1)(conv4)

    up5 = Conv2D(np.int(64 * n_filters_factor),
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(
                     size=(2, 2), interpolation='nearest')(bn4))
    merge5 = concatenate([bn1, up5], axis=3)
    conv5 = Conv2D(np.int(64 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge5)
    conv5 = Conv2D(np.int(64 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv5)
    bn5 = BatchNormalization(axis=-1)(conv5)

    final_layer_logits = [(Conv2D(n_output_classes, 1,
                                  activation='linear')(conv5))
                          for i in range(n_forecast_months)]
    final_layer_logits = tf.stack(final_layer_logits, axis=1)

    model = Model(inputs, final_layer_logits)
    model.summary()
    return model


def unet9(
          filter_size=3,
          n_filters_factor=1,
          n_forecast_months=7,
          n_output_classes=1):

    inputs = Input(shape=(112, 112, 8))

    conv1 = Conv2D(np.int(64 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(np.int(64 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv1)
    bn1 = BatchNormalization(axis=-1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(np.int(128 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(np.int(128 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv2)
    bn2 = BatchNormalization(axis=-1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(np.int(256 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(np.int(256 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv3)
    bn3 = BatchNormalization(axis=-1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(np.int(256 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(np.int(256 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv4)
    bn4 = BatchNormalization(axis=-1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = Conv2D(np.int(512 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(np.int(512 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv5)
    bn5 = BatchNormalization(axis=-1)(conv5)

    up6 = Conv2D(np.int(256 * n_filters_factor),
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(
                     size=(2, 2), interpolation='nearest')(bn5))
    merge6 = concatenate([bn4, up6], axis=3)
    conv6 = Conv2D(np.int(256 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(np.int(256 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv6)
    bn6 = BatchNormalization(axis=-1)(conv6)

    up7 = Conv2D(np.int(256 * n_filters_factor),
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(
                     size=(2, 2), interpolation='nearest')(bn6))
    merge7 = concatenate([bn3, up7], axis=3)
    conv7 = Conv2D(np.int(256 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(np.int(256 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv7)
    bn7 = BatchNormalization(axis=-1)(conv7)

    up8 = Conv2D(np.int(128 * n_filters_factor),
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(
                     size=(2, 2), interpolation='nearest')(bn7))
    merge8 = concatenate([bn2, up8], axis=3)
    conv8 = Conv2D(np.int(128 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(np.int(128 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv8)
    bn8 = BatchNormalization(axis=-1)(conv8)

    up9 = Conv2D(np.int(64 * n_filters_factor),
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(
                     size=(2, 2), interpolation='nearest')(bn8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(np.int(64 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(np.int(64 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(np.int(64 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv9)

    final_layer_logits = [(Conv2D(n_output_classes, 1,
                                  activation='linear')(conv9))
                          for i in range(n_forecast_months)]
    #out = final_layer_logits#[:, tf.newaxis]
    #out = tf.transpose(out, [0, 4, 2, 3, 1])
    final_layer_logits = tf.stack(final_layer_logits, axis=1)

    model = Model(inputs, final_layer_logits)

    model.summary()
    return model


if __name__ == '__main__':
    unet5(input_shape=(112, 112, 7))
