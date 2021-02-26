# -----------------------------------------------------------------------------
#                                Train module                                 #
# -----------------------------------------------------------------------------
# author: Lu Li                                                               #
# mail: lilu35@mail2.sysu.edu.cn                                              #
# -----------------------------------------------------------------------------
# This Repo is train module of artifical intelligence models, it has three    #
# types class, train class for machine learning models using skcit-learn libs,#
# deep learning models using keras libs in tensorflow (high-level API) and    #
# other models using gradientTape def in tensorflow libs. Notability, all     #
# classes are limited interface based on abstractmethods class in base.py     #
# -----------------------------------------------------------------------------


import os
import time

import numpy as np
import tensorflow as tf


def get_callback(save_path, task):
    # mkdir
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # checkpoint file path
    checkpoint_path = save_path + str(task) + '/' + str(task) + '.ckpt'
    # callback
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            save_weights_only=True,
            monitor='loss'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss')
        # monitor='')
    ]
    return callbacks


class Trainer():

    def __init__(self):
        pass

    def 

