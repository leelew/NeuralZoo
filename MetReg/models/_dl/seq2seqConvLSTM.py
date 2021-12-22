# ==============================================================================
# ConvLSTM in seq2seq mode
#
# (1) simple:
# (2) teacher forcing:
#
# author: Lu Li
# email: lilu83@mail.sysu.edu.cn
# ==============================================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, Input
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.convolutional_recurrent import \
    ConvLSTM2DCell

from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Dense


#TODO: 把训练方式作为形参设置，整合iterative/teacher forcing/standard
class IterativeConvLSTM2D(layers.Layer):
    def __init__(self, 
                 filters, 
                 kernel_size, 
                 padding='valid', 
                 strides=1, 
                 return_sequences=True, 
                 return_states=True):
        self.filters=filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.return_sequences = return_sequences
        self.return_states = return_states
        super().__init__()

    def build(self, input_shape):

        self.t = input_shape[1]
        print(self.t)
        
        self.cell_list = []
        for i in range(self.t):
            self.cell_list.append(ConvLSTM2DCell(filters=self.filters, 
                       kernel_size=self.kernel_size,
                       padding=self.padding, 
                       strides=self.strides))
        self.built = True

    def call(self, inputs, initial_state):
      
        if initial_state is not None: h, c = initial_state

        h_list = []
        for i in range(self.t):
            h, [h, c] = self.cell_list[i](h, states=[h, c])
            h_list.append(h)

        return tf.stack(h_list, axis=1)
        



class __Seq2seqConvLSTM(layers.Layer):
    def __init__(self,
                 n_filters_factor,
                 filter_size,
                 dec_len,
                 train_mode=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dec_len = dec_len
        self.train_mode = 'teacher'

        # encode convlstm
        self.enc_convlstm = layers.ConvLSTM2D(np.int(16 * n_filters_factor),
                                              kernel_size=filter_size,
                                              return_sequences=True,
                                              return_state=True,
                                              padding='same',
                                              kernel_initializer='he_normal')
        self.enc_bn = layers.BatchNormalization(axis=-1)

        # decode convlstm
        self.dec_convlstm = layers.ConvLSTM2D(np.int(16 * n_filters_factor),
                                              kernel_size=filter_size,
                                              return_sequences=True,
                                              padding='same',
                                              kernel_initializer='he_normal')
        self.dec_bn = layers.BatchNormalization(axis=-1)

        #
        self.dec_convlstm_cell = ConvLSTM2DCell(np.int(16 * n_filters_factor),
                                                kernel_size=filter_size,
                                                padding='same',
                                                kernel_initializer='he_normal')

    def call(self, x_encoder, x_decoder, training=None, mask=None):
        print(x_encoder.shape)
        print(x_decoder.shape)

        enc_out, state = self.encoder(x_encoder=x_encoder)
        print(enc_out.shape)

        if self.train_mode == 'teacher':
            if x_decoder:
                raise KeyError('teacher forcing training needs decoder inputs')
            dec_out = self.decoder(x_decoder, initial_state=state)
            print(dec_out.shape)
        else:
            dec_out = self.decoder(enc_out[:, -1], initial_state=state)
        return dec_out

    def encoder(self, x_encoder):
        enc_out, state_h, state_c = self.enc_convlstm(x_encoder)
        return enc_out, [state_h, state_c]

    def decoder(self, initial_x_decoder, initial_state=None):
        """decoder follow standard propagation method."""
        print('1')
        print(initial_x_decoder.shape)
      
        h, c = initial_state
        #x = initial_x_decoder
        print(h.shape)
        x_stack, h_stack, c_stack = [], [], []
        for i in range(self.dec_len):
            print(i)
            x = initial_x_decoder[:,i]
            print(x.shape)
            x, [h, c] = self.dec_convlstm_cell(initial_x_decoder=x,
                                               initial_state=[h, c])
            x_stack.append(x)

        return x


class Seq2seqConvLSTM(Model):
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

    def fit(self, X, y, batch_size, epochs, validation_data, callbacks):
        mdl = self.train_model()
        mdl.fit(X, y, batch_size, epochs, validation_data, callbacks)
        return self

    def train_model(self):
        X_encoder = Input((7, 112, 112, 1))
        X_decoder = Input((7, 112, 112, 1))

        encoder_outputs, encoder_states = self.encoder_convlstm(X_encoder)
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
        encoder_outputs, encoder_states = self.encoder_convlstm(X_encoder)
        decoder_input = encoder_states[-1]

        for _ in range(7):
            decoder_inputs, encoder_states = self.decoder_convlstm_cell(
                decoder_input, states=encoder_states)
            y_pred.append(decoder_inputs)
        
        return tf.stack(y_pred, axis=1)




class Seq2seqConvLSTM(layers.Layer):
    def __init__(self,
                 n_filters_factor,
                 filter_size,
                 dec_len,
                 train_mode=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dec_len = dec_len
        self.train_mode = 'teacher'

        # encode convlstm
        self.enc_convlstm = layers.ConvLSTM2D(np.int(16 * n_filters_factor),
                                              kernel_size=filter_size,
                                              return_sequences=True,
                                              return_state=True,
                                              padding='same',
                                              kernel_initializer='he_normal')
        self.enc_bn = layers.BatchNormalization(axis=-1)

        # decode convlstm
        self.dec_convlstm = layers.ConvLSTM2D(np.int(16 * n_filters_factor),
                                              kernel_size=filter_size,
                                              return_sequences=True,
                                              padding='same',
                                              kernel_initializer='he_normal')
        self.dec_bn = layers.BatchNormalization(axis=-1)

        #
        self.dec_convlstm_cell = ConvLSTM2DCell(np.int(16 * n_filters_factor),
                                                kernel_size=filter_size,
                                                padding='same',
                                                kernel_initializer='he_normal')

    def call(self, x_encoder, x_decoder, training=None, mask=None):
        print(x_encoder.shape)
        print(x_decoder.shape)

        enc_out, state = self.encoder(x_encoder=x_encoder)
        print(enc_out.shape)

        if self.train_mode == 'teacher':
            if x_decoder:
                raise KeyError('teacher forcing training needs decoder inputs')
            dec_out = self.decoder(x_decoder, initial_state=state)
            print(dec_out.shape)
        else:
            dec_out = self.decoder(enc_out[:, -1], initial_state=state)
        return dec_out

    def encoder(self, x_encoder):
        enc_out, state_h, state_c = self.enc_convlstm(x_encoder)
        return enc_out, [state_h, state_c]

    def decoder(self, initial_x_decoder, initial_state=None):
        """decoder follow standard propagation method."""
        print('1')
        print(initial_x_decoder.shape)
      
        h, c = initial_state
        #x = initial_x_decoder
        print(h.shape)
        x_stack, h_stack, c_stack = [], [], []
        for i in range(self.dec_len):
            print(i)
            x = initial_x_decoder[:,i]
            print(x.shape)
            x, [h, c] = self.dec_convlstm_cell(initial_x_decoder=x,
                                               initial_state=[h, c])
            x_stack.append(x)

        return x