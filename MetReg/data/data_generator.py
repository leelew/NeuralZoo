import numpy as np
from sklearn.model_selection import train_test_split


class Data_generator():
    """generate dataset pipeline for prediction application.

    A class ensemble data loading, preprocessing, validation and generating 
    process, which create [ND.ARRAY] type of dataset pipeline. 

    .. Note: to satisfy all machine learning models, the shape of dataset 
             pipeline is (samples, timesteps[in], height, width, features) for
             features dataset, and (sample, height, width) for one-step 
             prediction label or (sample, height, width, timesteps[out]) for
             multi-step prediction labels. 

    .. Note: for deep learning prediction, tensorflow dataset pipeline should
             be used after generating dataset from `Data_generator`. pls refer
             to `MetReg.data.data_loader` class to implement same action with
             `pytorch.utils.DataLoader`.  

    Args:

    Returns:

    Raises:
    """

    def __init__(self,
                 X,
                 y=None,
                 train_valid_ratio=0.2,
                 len_inputs=10,
                 len_output=1,
                 window_size=7):
        self.X = X
        self.y = y

        self.train_valid_ratio = train_valid_ratio
        self.len_inputs = len_inputs
        self.len_output = len_output
        self.window_size = window_size

    def _split_train_valid_order(self, inputs):
        """Split data into train and valid dataset by order"""
        # get train sets length
        train, valid = train_test_split(inputs,
                                        test_size=self.train_valid_ratio,
                                        shuffle=False)
        return train, valid

    def _get_batch_idx(self,
                       timesteps=None,
                       len_inputs=10,
                       len_output=1,
                       window_size=7):
        # caculate the last time point to generate batch
        end_idx = timesteps - len_inputs - len_output - window_size
        # generate index of batch start point in order
        batch_start_idx = range(end_idx)
        # generate inputs idx
        input_batch_idx = [(range(i, i + len_inputs)) for i in batch_start_idx]
        # generate outputs
        output_batch_idx = [(range(i + len_inputs + window_size,
                                   i + len_inputs + window_size + len_output))
                            for i in batch_start_idx]
        return input_batch_idx, output_batch_idx, len(batch_start_idx)

    def _get_batch_data(self, X, y):
        """Generate inputs and outputs for SMNET."""
        # generate batch index
        input_batch_idx, output_batch_idx, num_samples = self._get_batch_idx(
            timesteps=X.shape[0], 
            len_inputs=self.len_inputs,
            len_output=self.len_output,
            window_size=self.window_size,)
        # generate inputs
        inputs = np.take(X, input_batch_idx, axis=0). \
            reshape(num_samples, self.len_inputs,
                    X.shape[1], X.shape[2], X.shape[3])
        # generate outputs
        outputs = np.take(y, output_batch_idx, axis=0). \
            reshape(num_samples, self.len_output,
                    y.shape[1], y.shape[2], 1)
        return inputs, outputs

    def __call__(self):
        # split data
        X_train, X_valid = self._split_train_valid_order(self.X)
        y_train, y_valid = self._split_train_valid_order(self.y)
        print(X_train.shape)
        print(y_train.shape)

        # generate batch data
        X_train, y_train = self._get_batch_data(X_train, y_train)
        X_valid, y_valid = self._get_batch_data(X_valid, y_valid)
        print(X_train.shape)
        print(y_train.shape)

        # package into dict
        data = dict()
        data['X_train'] = X_train
        data['X_valid'] = X_valid
        data['y_train'] = y_train
        data['y_valid'] = y_valid

        return data


