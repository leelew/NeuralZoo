import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class SSIM(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return -1*tf.image.ssim(
    img1=y_pred, img2=y_true, max_val=1, filter_size=3, filter_sigma=1.5, k1=0.01, k2=0.03
)



OUTPUT_CHANNELS = 1


def downsample(filters, size, apply_batchnorm=True,apply_pooling=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    # initializer = tf.keras.initializers.he_normal()

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())
    if apply_pooling:
        result.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

    # result.add(DCTLayer(6))
    return result


down_model = downsample(64, 4)
down_result = down_model(tf.expand_dims(inp, 0))
print(down_result.shape)


def upsample(filters, size, apply_dropout=False, apply_upsampling=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    # initializer = tf.keras.initializers.he_normal()
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))  #Transpose

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())
    if apply_upsampling:
        result.add(tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest'))
    # result.add(DCTLayer(7))
    return result


# @tf.function
def MSDM():
  inputs = tf.keras.layers.Input(shape=[256,256,2])


  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4,apply_pooling=False), # (bs, 64, 64, 128)
    downsample(256, 4,apply_pooling=False), # (bs, 32, 32, 256)
    downsample(512, 2), # (bs, 16, 16, 512)
    downsample(512, 2), # (bs, 8, 8, 512)
    downsample(512, 2), # (bs, 4, 4, 512)
    downsample(512, 2), # (bs, 2, 2, 512)
    downsample(512, 2), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 2, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 2, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 2, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 2), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4,apply_upsampling=False), # (bs, 64, 64, 256)
    upsample(64, 4,apply_upsampling=False), # (bs, 128, 128, 128)
  ]


  initializer = tf.random_normal_initializer(0., 0.02)
  # initializer = tf.keras.initializers.he_normal()
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')

  x = inputs

  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)
  skips = reversed(skips[:-1])



  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])


  z= last(x)
  return tf.keras.Model(inputs=inputs, outputs=z) #,inputs2]



MSDM = MSDM()
MSDM.summary()
