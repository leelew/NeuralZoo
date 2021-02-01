from autosklearn import regression


def automl():
    mdl = regression.AutoSklearnRegressor()
    return mdl
