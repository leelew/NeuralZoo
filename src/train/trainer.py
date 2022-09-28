import sys

sys.path.append('../../src/')

import json

import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from data.data_generator import DataLoader
from factory.callback import CallBacks
from model.convlstm_factory import SMNet
from factory.loss import MaskMSELoss, MaskSSIMLoss
from model.gan_factory import GAN

def train(X_l3,
          X_l4,
          y,
          land_mask,
          id,
          do_transfer_learning,
          learning_rate,
          n_filters_factor,
          filter_size,
          batch_size,
          epochs,
          model_name):
    # wandb setting
    default = dict(learning_rate=learning_rate,
                   n_filters_factor=n_filters_factor,
                   filter_size=filter_size,
                   batch_size=batch_size,
                   epochs=epochs)
    wandb.init(config=default, allow_val_change=True)


    # model
    if model_name == 'smnet': 
        model = SMNet()
        # compile
        model.compile(optimizer=Adam(wandb.config.learning_rate), loss=MaskMSELoss(land_mask))

    elif model_name == 'gan': model = GAN()


    # fit
    x_train_l3, x_valid_l3, x_test_l3 = X_l3
    x_train_l4, x_valid_l4, x_test_l4 = X_l4
    y_train, y_valid, y_test = y

    #FIXME: Reframe the model, compile, fit process
    if model_name == 'gan':
        model.fit((x_train_l3, x_train_l4), y_train,
                   batch_size=wandb.config.batch_size,
                   epochs=wandb.config.epochs)
    else:
        model.fit([x_train_l3, x_train_l4], y_train,
                batch_size=wandb.config.batch_size,
                epochs=wandb.config.epochs,
                callbacks=CallBacks()(),
                validation_data=([x_valid_l3, x_valid_l4], y_valid))
    
    # predict
    model.load_weights('/hard/lilu/Checkpoints/') 
    y_test_predict = model.predict([x_test_l3, x_test_l4])
    np.save('/hard/lilu/y_test_pred_{}'.format(id), y_test_predict)
    np.save('/hard/lilu/y_test_obs_{}'.format(id), y_test)


if __name__ == '__main__':
    train()