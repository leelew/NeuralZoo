from sklearn import gaussian_process


class GP:

    def __init__(self,
                 kernel=None,
                 ):
        self.kernel = kernel

    def __call__(self):
        mdl = gaussian_process.GaussianProcessRegressor(
            kernel=self.kernel)
        return mdl
