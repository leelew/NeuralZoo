import six
import abc

@six.add_metaclass(abc.ABCMeta)
class BaseModel():

    def __init__(self):
        self.regressor = None

    @abc.abstractmethod
    def fit(self, X, y):
        """Ensure all Machine learning model have `fit` methods"""
        pass

    def predict(self, X):
        """Predict for all Sklearn type models"""
        if self.regressor is None:
            raise NotImplementedError('Fit model before predict!')
        else:
            return self.regressor.predict(X)

    def get_hyperparameters_search_space(self): 
        #todo: autoML
        pass
