
import sys

sys.path.append('../')
import numpy as np
from MetReg.models.ml.elm import ExtremeLearningRegressor
from MetReg.models.ml.gp import GaussianProcessRegressor
from MetReg.models.ml.knn import KNNRegressor
from MetReg.models.ml.linear import (BaseLinearRegressor, ElasticRegressor,
                                     ExpandLinearRegressor, LassoRegressor,
                                     RidgeRegressor)
from MetReg.models.ml.mlp import MLPRegressor
from MetReg.models.ml.svr import LinearSVRegressor, SVRegressor
from MetReg.models.ml.tree import (AdaptiveBoostingRegressor,
                                   BaseTreeRegressor, ExtraTreesRegressor,
                                   ExtremeGradientBoostingRegressor,
                                   GradientBoostingRegressor,
                                   LightGradientBoostingRegressor,
                                   RandomForestRegressor)


def test_linear_regressors():
    X = np.random.random(size=(100, 3))
    y = np.random.random(size=(100,))

    BLR = BaseLinearRegressor()
    BLR.fit(X, y)
    BLR.predict(X)

    RR = RidgeRegressor()
    RR.fit(X, y)
    RR.predict(X)

    LR = LassoRegressor()
    LR.fit(X, y)
    LR.predict(X)

    ER = ElasticRegressor()
    ER.fit(X, y)
    ER.predict(X)


def test_tree_regressors():
    X = np.random.random(size=(100, 3))
    y = np.random.random(size=(100,))

    BTR = BaseTreeRegressor()
    BTR.fit(X, y)
    BTR.predict(X)

    RFR = RandomForestRegressor()
    RFR.fit(X, y)
    RFR.predict(X)

    ETR = ExtraTreesRegressor()
    ETR.fit(X, y)
    ETR.predict(X)

    ABR = AdaptiveBoostingRegressor()
    ABR.fit(X, y)
    ABR.predict(X)

    GBR = GradientBoostingRegressor()
    GBR.fit(X, y)
    GBR.predict(X)

    EGBR = ExtremeGradientBoostingRegressor()
    EGBR.fit(X, y)
    EGBR.predict(X)

    LGBR = LightGradientBoostingRegressor()
    LGBR.fit(X, y)
    LGBR.predict(X)


def test_svr_regressors():
    X = np.random.random(size=(100, 3))
    y = np.random.random(size=(100,))

    LSVR = LinearSVRegressor()
    LSVR.fit(X, y)
    LSVR.predict(X)

    SVR = SVRegressor()
    SVR.fit(X, y)
    SVR.predict(X)


def test_knn_regressor():
    X = np.random.random(size=(100, 3))
    y = np.random.random(size=(100,))

    KR = KNNRegressor()
    KR.fit(X, y)
    KR.predict(X)

def test_mlp_regressor():

    X = np.random.random(size=(100, 3))
    y = np.random.random(size=(100,))

    MR = MLPRegressor()
    MR.fit(X, y)
    MR.predict(X)

def test_gp_regressor():

    X = np.random.random(size=(100, 3))
    y = np.random.random(size=(100,))

    GPR = GaussianProcessRegressor()
    GPR.fit(X, y)
    GPR.predict(X)


def test_elm_regressor():

    X = np.random.random(size=(100, 3))
    y = np.random.random(size=(100,))

    GPR = ExtremeLearningRegressor()
    GPR.fit(X, y)
    GPR.predict(X)

if __name__ == '__main__':
    #test_linear_regressors()
    #test_tree_regressors()
    test_svr_regressors()
    test_knn_regressor()
    test_mlp_regressor()
    test_gp_regressor()
    test_elm_regressor()
