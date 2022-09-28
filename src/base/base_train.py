# -----------------------------------------------------------------------------
#                                Base module                                  #
# -----------------------------------------------------------------------------
# author: Lu Li                                                               #
# mail: lilu35@mail2.sysu.edu.cn                                              #
# -----------------------------------------------------------------------------
# This Repo provides base classes for training class, include Base_train_ml   #
# for machine learning (sklearn API), Base_train_keras for deep learning      #
# (keras API), Base_train_tf for deep learning own-build (low-level API)      #
# -----------------------------------------------------------------------------

# TODO: Need Improved!


import six
import abc


@six.add_metaclass(abc.ABCMeta)
class Base_train_keras():

    def __init__(self, x_train, y_train, mdl, config=None):
        """This class provide different training methods corresponding to
        different models categories. 

        Args:
            [T] mode:
            x_train (tf.data):
                shape of (batch_size, timestep, feature)
            y_train (tf.data):
                shape of (batch_size, )

            [S] mode:
            x_train (tf.data):
                shape of (batch_size, height, width, feature)
            y_train (tf.data):
                shape of (batch_size, height, width)

            [ST] mode:
            x_train (tf.data):
                shape of (batch_size, timestep, height, width, feature)
            y_train (tf.data):
                shape of (batch_size, height, width)

            mdl (keras class):
                model class for own-choice setting from config
            config (dict, optional): 
                configure dict for training parameters. Defaults to None.
                if set to None, means use default setting.
        """
        pass

    @abc.abstractmethod
    def _check_data_get_mode(self, x_train, y_train):
        """check dl data and setting mode.
        """
        return

    @abc.abstractmethod
    def _get_data_attr(self):
        """get attribute of dataset corresponding different dimensions.
        """
        return

    def callback(self):
        pass

    @abc.abstractmethod
    def train_global(self, mdl, x_train, y_train):
        return
    
    @abc.abstractmethod
    def train_grid(self, mdl, x_train, y_valid):
        return

    def __repr__(self):
        return '[INFO] This model is trained by high level API !'


@six.add_metaclass(abc.ABCMeta)
class Base_train_tf():

    def __init__(self, optimizers, epoch):
        self.opt = optimizers
        self.epoch = epoch

    @abc.abstractmethod
    def train_one_step(self):
        return

    @abc.abstractmethod
    def train(self):
        return

    @abc.abstractmethod
    def train_multigrids(self):
        return

    def __repr__(self):
        return '[INFO] This model is trained by low-level API !'


@six.add_metaclass(abc.ABCMeta)
class Base_train_ml():

    def __init__(self, x_train, y_train, model_name: list):
        """
        This train class provide two training mode. list as:

        [global] mode: 
        if inputs is 2D dataset (samples, feature), generate 
        global model, i.e., generate one model (set of parameters) 
        among multigrids (also could used for single point model).

        [grid] mode:
        if inputs is 4D dataset (samples, height, width, feature),
        generate grids model, i.e., generate height*width models 
        among all grids. 

        Returns:
            [global] mode: single dict shape of {number of models},
                           the key is from model_name list, and the
                           value is correponding trained model.

            [grid] mode: nest dict shape of {number of models},
                         the key is from model_name list, and the
                         value is correponding nest list (2D) shape
                         as (height, width).
        """
        pass

    def _check_data_get_mode(self):
        """check dimension of input data and select mode. This
        train class only support 2D and 4D inputs and support 
        corresponding two training modes.
        """
        return

    def _get_data_attr(self):
        """get data attribute (shape) of each training mode. must
        exec after _check_data_get_mode() func.
        """
        return

    @staticmethod
    @abc.abstractmethod
    def _train_grid(reg, x_train, y_train):
        """train on one grid, this func is static and unchanged.

        Args:
            reg (class): 
                untrained regression class generate from sklearn.
            x_train (numpy array):
                numpy array of shape (timestep, feature)
            y_train (numpy array): 
                numpy array of shape (timestep,)
        """
        return

    @abc.abstractmethod
    def global_train_models(self, x_train, y_train):
        """[global] mode. train globally using different models"""
        return

    @abc.abstractmethod
    def _train_grids(self, reg, x_train, y_train):
        """train on several grids. dimension is generated from 
        self input using _get_data_attr() func.

        Args:
            reg (class): 
                untrained regression class generate from sklearn.
            x_train (numpy array):
                numpy array of shape (timestep, lat, lon, feature)
            y_train ([type]): [description]
                numpy array of shape (timestep, lat, lon,)
        """
        return

    @abc.abstractmethod
    def grid_train_models(self, x_train, y_train):
        """[grid] mode. train on seperate grid using different mdls"""
        return

    def __repr__(self):
        return '[INFO] This model is trained by high-level SKLEARN API !'
