
from sklearn import svm


class svr:

    def __init__(self,
                 kernel='rbf',
                 C=0.6,
                 gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma

    def __call__(self):
        mdl = svm.SVR(
            kernel=self.kernel,
            gamma=self.gamma,
            C=self.C)
        return mdl
