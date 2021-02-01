#from MetReg.base.base_train import Base_train_ml


class train_ml():

    def __init__(self,
                 mdl,
                 X,
                 y=None,):
        self.mdl = mdl
        self.X = X
        self.y = y

    @staticmethod
    def _fit_grid(mdl, X, y):
        if mdl is None:
            raise KeyError('give a model class')
        else:
            mdl.fit(X, y)
        return mdl
