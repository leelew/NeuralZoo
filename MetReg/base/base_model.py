import six
import abc


@six.add_metaclass(abc.ABCMeta)
class BaseModel():

    def __init__(self):
        self.regressor = None

    @abc.abstractmethod
    def fit(self, X, y): pass

    def predict(self, X):
        if self.regressor is None:
            raise NotImplementedError('fit model before!')
        else:
            return self.regressor.predict(X)

    def __repr__(self): pass

    def get_hyperparameters_search_space(self): pass
