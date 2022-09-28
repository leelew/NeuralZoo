
import numpy as np
import sklearn.preprocessing as prep


class Data_preprocessor():
    """a class for preprocessing data.

    Implement basic preprocessing operation, including normalization, 
    interplot, outlier preprocessing, wavelet denoise.
    # (TODO)@lilu: outlier preprocess
    # (TODO)@lilu: wavelet denoise process

    .. rubic:: process loop
                0. check X, y
                1. concat X,y for process
                2. split dataset by time order
                3. normalization (turn default value to nan)
                4. interplot (first spatial, second temporal)
                5. separate into inputs and outputs

    .. Notes: exec only support special dataset frame, such as nd.array,
              satisfy dimensions, etc.

    Args:
        interp (bool, optional): whether interpolate. Defaults to True.
        normalize (bool, optional): whether normalize. Defaults to True.
        feat_name (list, optional): 
            list contain name of features. 
            Defaults to ['precipitation','soil temperature','pressure'].
        fill_value (int, optional): Defaults to -9999. 
        (TODO)@lilu: try different fill value of different features
    """

    def __init__(self,
                 X,
                 y=None,
                 interp=True,
                 normalize=True,
                 fill_value=-9999,
                 feat_name=['precipitation'],
                 ):
        self.X, self.y = X, y
        self.fill_value = fill_value
        self.feat_name = feat_name
        self.normalize = normalize
        self.interp = interp

        # get config
        self.set_config()

    def _normalize(self, inputs):
        """Normalization data using MinMaxScaler

        .. Notes: Instead normalize on the whole data, we first normalize
                  train data, and then normalize valid data by trained scaler.
                  we believe this is a correct operation to avoid introducing
                  test dataset information before benchmark process. Therefore,
                  the input of this func must have train & valid datasets.
        """
        return prep.MinMaxScaler().fit_transform(inputs)

    def _interp(self, inputs):
        """Interplot using mean value.

        Args:
            inputs (nd.array):
                must be timeseries or images.
                array shape of (timestep,) or (height, width)
        """
        if inputs.ndim == 1:
            return np.nan_to_num(inputs, nan=np.nanmean(inputs),)
        elif inputs.ndim == 2:
            H, W = inputs.shape
            inputs = np.nan_to_num(inputs.reshape(-1, 1),
                                   nan=np.nanmean(inputs))
            return inputs.reshape(H, W)

    def _denoise(self, inputs, flag_features=False):
        """denoise of inputs, only for features.

        Args:
            inputs ([type]): [description]
            flag_features (bool, optional):
                flag for identify features and label.
        """
        if not flag_features:
            raise KeyError('wavelet denoise method not support for label.')
        else:
            pass

    def _cluster(self):
        pass

    def __call__(self):
        """Exec preprocessing process"""
        # (TODO)@lilu: check X,y

        # concat X,y for process
        data = np.concatenate([self.X, self.y], axis=-1)

        # turn fill value to nan
        data[data == self.fill_value] = np.nan

        # interplot
        data = data.reshape(-1, self.height, self.width)
        for i in range(data.shape[0]):
            data[i, :, :] = self._interp(data[i, :, :])
        data = data.reshape(-1, self.height, self.width, self.feat_size+1)

        # normalization
        if self.normalize:
            for i in range(self.height):
                for j in range(self.width):
                    data[:, i, j, :] = self._normalize(data[:, i, j, :])

        # separate
        X, y = data[:, :, :, :-1], data[:, :, :, -1]
        return X, y

    def __repr__(self):
        return str(self.config)

    def set_config(self):
        """Set configuration of class."""
        self.N, self.height, self.width, self.feat_size = self.X.shape

        self.config = {
            'name': 'auto processor',
            'timestep': self.N,
            'height': self.height,
            'width': self.width,
            'feat_size': self.feat_size,
            'feat_name': self.feat_name,
            'interp': self.interp,
            'interp_method': 'mean value',
            'normalize': self.normalize,
            'normalize_method': 'min max scaler',
            'fill_value': self.fill_value
        }


if __name__ == "__main__":
    X = np.random.random(size=(100, 10, 10, 2))
    y = np.random.random(size=(100, 10, 10, 1))
    dp = Data_preprocessor(X, y)
    print(dp)
