import tensorflow as tf
from MetReg.base.base_model import BaseModel
from tensorflow.keras import models, Model
from tensorflow.keras import layers


class DNNRegressor(Model):

    def __init__(self,
                 activation='relu',):
        super().__init__()
        self.regressor = None
        self.dense1 = layers.Dense(16)
        self.dense2 = layers.Dense(8)
        self.dense3 = layers.Dense(1)

    def call(self, inputs):

        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
