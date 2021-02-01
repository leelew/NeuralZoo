
from sklearn import svm


class svr:

    def __init__(self,
                 kernel='rbf',
                 C=0.6,
                 gamma='scale',
                 
                 ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.regressor = None


    def fit(self, X, y):


        self.regressor = svm.SVR(
            kernel=self.kernel,
            gamma=self.gamma,
            C=self.C)

        self.regressor.fit(X, y)
        return self
