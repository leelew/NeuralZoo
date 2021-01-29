from MetReg.base.base_train import Base_train_ml


class train_ml():

    def __init__(self,
                 mdl,
                 X,
                 y=None,):
        self.mdl = mdl
        self.X = X
        self.y = y

    def __call__(self):
        self.mdl.fit(self.X, self.y)
        return self.mdl
