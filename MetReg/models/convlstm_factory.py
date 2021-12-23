import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, backend, layers
from tensorflow.keras.layers import (Add, BatchNormalization, Conv2D,
                                     ConvLSTM2D, Dense, Dropout,
                                     GlobalAveragePooling2D, Input, Lambda,
                                     MaxPooling2D, ReLU, Reshape, UpSampling2D,
                                     concatenate, multiply)
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.convolutional_recurrent import ConvLSTM2DCell


class ScheduleSamplingSeq2SeqConvLSTM():
    def __init__(self) -> None:
        pass

    def build():
        pass

    def compile():
        pass

    def fit():
        pass

    def load_weights():
        pass

    def predict():
        pass

    def reverse_schedule_sampling(self):
        pass

    def schedule_sampling(self, 
                          X_decoder, eta, itr, 
                          sampling_stop_iter=50000, 
                          sampling_changing_rate=0.00002, 
                          batch_size=2):
        # S, 7, 112, 112, 1
        zeros = np.zeros((2, 7, 112, 112, 1))

        if itr < sampling_stop_iter:
            eta -= sampling_changing_rate
        else:
            eta = 0.0

        random_flip = np.random.random_sample((2, 7))
        true_token = (random_flip < eta)
        ones = np.ones((112, 112, 1))
        zeros = np.zeros((112, 112, 1))
        real_input_flag = []
        for i in range(batch_size):
            for j in range(7):
                if true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
        real_input_flag = np.array(real_input_flag)
        real_input_flag = np.reshape(real_input_flag,
                                    (2, 7, 112, 112, 1))
        return eta, real_input_flag


class _Seq2seqConvLSTM():
    def __init__(self, 
                 filters, 
                 kernel_size, 
                 padding='same', 
                 kernel_initializer='he_normal', 
                 mode='train'):
        super().__init__()
    
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.kernel_initializer = kernel_initializer

        # encoder convlstm
        self.encoder_convlstm = ConvLSTM2D(filters=self.filters,
                                           kernel_size=self.kernel_size,
                                           return_sequences=True,
                                           return_state=True,
                                           padding=self.padding,
                                           kernel_initializer=self.kernel_initializer)
        self.encoder_bn = BatchNormalization()

        # decoder convlstm
        self.decoder_convlstm = ConvLSTM2D(filters=self.filters,
                                           kernel_size=self.kernel_size,
                                           return_sequences=True,
                                           return_state=True,
                                           padding=self.padding,
                                           kernel_initializer=self.kernel_initializer)
        self.decoder_bn = BatchNormalization()

        # decoder convlstm
        self.decoder_convlstm_cell = ConvLSTM2DCell(filters=self.filters,
                                                    kernel_size=self.kernel_size,
                                                    padding=self.padding,
                                                    kernel_initializer=self.kernel_initializer)
        self.decoder_bn = BatchNormalization()

        # dense
        self.dense = Dense(1)

    def compile(self, optimizer, loss):
        self.mdl = self.train_model()
        self.mdl.compile(optimizer=optimizer, loss=loss)

    def fit(self, X, y, batch_size, epochs, validation_data, callbacks):
        self.mdl.fit(X, y, 
                     batch_size=batch_size, 
                     epochs=epochs, 
                     validation_data=validation_data, 
                     callbacks=callbacks)

    def load_weights(self, filepath):
        self.mdl.load_weights(filepath)

    def train_model(self):
        X_encoder = Input((7, 112, 112, 1))
        X_decoder = Input((7, 112, 112, 1))

        encoder_outputs, state_h, state_c = self.encoder_convlstm(X_encoder)
        encoder_states = [state_h, state_c]
        print(encoder_outputs.shape)
        decoder_outputs = []
        for i in range(7):
            decoder_output, encoder_states = self.decoder_convlstm_cell(
                X_decoder[:,i], states=encoder_states)
            decoder_outputs.append(decoder_output)
        decoder_outputs = tf.stack(decoder_outputs, axis=1)
        outputs = tf.concat([encoder_outputs, decoder_outputs], axis=1)
        outputs = self.dense(outputs)
        mdl = Model([X_encoder, X_decoder], outputs)
        return mdl

    def predict(self, X_encoder):
        y_pred = []
        encoder_outputs, state_h, state_c = self.encoder_convlstm(X_encoder)
        decoder_input = state_h
        encoder_states = [state_h, state_c]

        for _ in range(7):
            decoder_inputs, encoder_states = self.decoder_convlstm_cell(
                decoder_input, states=encoder_states)
            
            y_pred.append(self.dense(decoder_inputs))
        
        return tf.stack(y_pred, axis=1)



def ConvLSTM():
    inputs = Input(shape=(7, 112, 112, 8))

    outputs, h, c = tf.keras.layers.ConvLSTM2D(16, 5, 
                        padding='same', 
                        kernel_initializer='he_normal', return_state=True)(inputs)
    out = Lambda(lambda x: backend.concatenate([x] * 7, axis=1))(h[:, tf.newaxis])
    out = ConvLSTM2D(16, 3, padding='same', return_sequences=True)(out)
    out = tf.keras.layers.Dense(1)(out)

    mdl = Model(inputs, out)
    mdl.summary()    
    return mdl

def trajGRU():
    inputs = Input(shape=(7, 112, 112, 8))
    inputs_dec = Input(shape=(7, 112, 112, 5))

    outputs, states = TrajGRU(num_filter=16, b_h_w=(None, 112, 112))(inputs)
    out, state = TrajGRU(num_filter=16, b_h_w=(None, 112, 112))(inputs_dec, states)
    out = tf.keras.layers.Dense(1)(out)

    mdl = Model([inputs, inputs_dec], out)
    mdl.summary()    
    return mdl

def SAConvLSTM():
    inputs = Input(shape=(7, 112, 112, 8))
    inputs_dec = Input(shape=(7, 112, 112, 5))

    outputs, states = SelfAttentionConvLSTM2D(16, 5, padding='same')(inputs)
    out, state = SelfAttentionConvLSTM2D(16, 3, padding='same')(inputs_dec, states)
    out = tf.keras.layers.Dense(1)(out)

    mdl = Model([inputs, inputs_dec], out)
    mdl.summary()    
    return mdl

#FIXME: Correct! Read original paper!
def predRNN():
    inputs = Input(shape=(7, 112, 112, 8))
    inputs_dec = Input(shape=(7, 112, 112, 5))

    outputs, states = predRNN(16, 5, padding='same')(inputs)
    out, state = predRNN(16, 3, padding='same')(inputs_dec, states)
    out = tf.keras.layers.Dense(1)(out)

    mdl = Model([inputs, inputs_dec], out)
    mdl.summary()    
    return mdl

def ef_convlstm():
    inputs = Input(shape=(7, 112, 112, 8))
    outputs = EF()(inputs)
    outputs = Dense(1)(outputs)
    mdl = Model(inputs, outputs)
    mdl.summary() 

    return mdl

def residual_convlstm():
    inputs = Input(shape=(7, 112, 112, 8))
    # preprocess l4
    outputs = tf.keras.layers.ConvLSTM2D(16, 5, 
                        return_sequences=True, 
                        padding='same', 
                        kernel_initializer='he_normal')(inputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.ReLU()(outputs)

    outputs = tf.keras.layers.Add()([Dense(16)(outputs), inputs])

    states = ConvLSTM2D(16, 3, padding='same', return_sequences=True)(outputs)
    states = tf.transpose(states, [0, 4, 2, 3, 1])
    states = Dense(1)(states)
    states = tf.transpose(states, [0, 4, 2, 3, 1])

    out = Lambda(lambda x: backend.concatenate([x] * 7, axis=1))(states)
    out = ConvLSTM2D(16, 3, padding='same', return_sequences=True)(out)
    out = tf.keras.layers.Dense(1)(out)

    mdl = Model(inputs, out)
    mdl.summary() 

    return mdl

def convlstm():
    inputs = Input(shape=(7, 112, 112, 8))
    # preprocess l4
    outputs = tf.keras.layers.ConvLSTM2D(16, 5, 
                        return_sequences=True, 
                        padding='same', 
                        kernel_initializer='he_normal')(inputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.ReLU()(outputs)

    states = ConvLSTM2D(16, 3, padding='same', return_sequences=True)(outputs)
    states = tf.transpose(states, [0, 4, 2, 3, 1])
    states = Dense(1)(states)
    states = tf.transpose(states, [0, 4, 2, 3, 1])

    out = Lambda(lambda x: backend.concatenate([x] * 7, axis=1))(states)
    out = ConvLSTM2D(16, 3, padding='same', return_sequences=True)(out)
    out = tf.keras.layers.Dense(1)(out)

    mdl = Model(inputs, out)
    mdl.summary()    
    return mdl

def convlstm_teacher_forcing():

    inputs = Input(shape=(7, 112, 112, 8))
    inputs_tf = Input(shape=(7, 112, 112, 5))

    # preprocess l4
    outputs = tf.keras.layers.ConvLSTM2D(16, 5, 
                        return_sequences=True, 
                        padding='same', 
                        kernel_initializer='he_normal')(inputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.ReLU()(outputs)

    outputs, h, c = ConvLSTM2D(16, 3, padding='same', return_state=True)(outputs)
    #states = tf.transpose(states, [0, 4, 2, 3, 1])
    #states = Dense(1)(states)
    #states = tf.transpose(states, [0, 4, 2, 3, 1])
    print('1')
    print(outputs.shape)



    out = ConvLSTM2D(16, 3, padding='same', return_sequences=True)(inputs_tf, initial_state=[h, c])
    out = tf.keras.layers.Dense(1)(out)

    mdl = Model([inputs, inputs_tf], out)

    mdl.summary()    
    return mdl

def convlstm_iterative():
    inputs = Input(shape=(7, 112, 112, 8))

    # preprocess l4
    outputs = tf.keras.layers.ConvLSTM2D(16, 5, 
                        return_sequences=True, 
                        padding='same', 
                        kernel_initializer='he_normal')(inputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.ReLU()(outputs)

    outputs, h, c = ConvLSTM2D(16, 3, padding='same', return_state=True)(outputs)
    #states = tf.transpose(states, [0, 4, 2, 3, 1])
    #states = Dense(1)(states)
    #states = tf.transpose(states, [0, 4, 2, 3, 1])

    out = IterativeConvLSTM2D(16, 3, padding='same', return_sequences=True)(inputs, initial_state=[h, c])
    out = tf.keras.layers.Dense(1)(out)

    mdl = Model(inputs, out)

    mdl.summary()    
    return mdl



def base_model():
    inputs = Input(shape=(7, 112, 112, 8))
    out = tf.keras.layers.ConvLSTM2D(8, 5, 
                        return_sequences=True, 
                        padding='same', 
                        kernel_initializer='he_normal')(inputs)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    mdl = Model(inputs, out)

    return mdl


def SMNet():
    # inputs 
    in_l3 = Input(shape=(7, 112, 112, 1))
    in_l4 = Input(shape=(7, 112, 112, 8))
    
    # preprocess l3
    #out_l3 = DIConvLSTM2D.DIConvLSTM(filters=8, kernel_size=5)(in_l3)
    #out_l3 = tf.keras.layers.BatchNormalization()(out_l3)
    #out_l3 = tf.keras.layers.ReLU()(out_l3)

    # preprocess l4
    out_l4 = tf.keras.layers.ConvLSTM2D(8, 5, 
                        return_sequences=True, 
                        padding='same', 
                        kernel_initializer='he_normal')(in_l4)
    out_l4 = tf.keras.layers.BatchNormalization()(out_l4)
    out_l4 = tf.keras.layers.ReLU()(out_l4)

    out = tf.keras.layers.Add()([out_l4, in_l4])

    states = ConvLSTM2D(16, 3, padding='same', return_sequences=False)(out)
    #states = tf.transpose(states, [0, 4, 2, 3, 1])
    #states = Dense(1)(states)
    #states = tf.transpose(states, [0, 4, 2, 3, 1])

    out = Lambda(lambda x: backend.concatenate([x[:, tf.newaxis]] * 7, axis=1))(states)
    out = ConvLSTM2D(16, 3, padding='same', return_sequences=True)(out)
    out = tf.keras.layers.Dense(1)(out)

    mdl = Model([in_l3, in_l4], out)
    mdl.summary()

    return mdl
