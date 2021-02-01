# -----------------------------------------------------------------------------
#                    Machine Learning Model Repository (MLMR)                 #
# -----------------------------------------------------------------------------
# author: Lu Li                                                               #
# mail: lilu35@mail2.sysu.edu.cn                                              #
# -----------------------------------------------------------------------------
# This Repo contains nearly all Machine Learning regression models, which is  #
# implemented by using skcit-learn library (https://scikit-learn.org/stable/  #
# modules/linear_model.html#). Additinal infos please see guide.pdf           #
# -----------------------------------------------------------------------------


import sys

import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import xgboost
from sklearn import (ensemble, gaussian_process, linear_model, neural_network,
                     svm, tree)
from sklearn.model_selection import GridSearchCV


class Linear_Regression():
    """ Implementation of all linear regression methods. 

    Models in this class include:

    default regression: 
    NOTE: the difference of these models is loss function.
    1. default linear regression (optimized by OLS)
    2. ridge regression (default/CV)
    3. lasso regression (default/CV/LAR/LARCV/LARIC)
    4. elastic-net regression (default/CV)

    estimate regression:
    NOTE: this method have different optimize methods with OLS.
    5. automatic_relevance_determination
    6. stochastic gradient descent method
    7. online passive-aggressive

    robust regression:
    8. RANSAC
    9. Theil-Sen
    10. Huber
    """

    def __init__(self): pass

    def default(self):
        """default linear regression."""
        reg = linear_model.LinearRegression()
        return reg

    def ridge(self,
              CV=False, cv_range=None, cv_fold=None):
        """ridge regression, i.e., loss function + L2 regularization

        NOTE: CV for alpha, i.e., coefficient of L2, cv_range > 0
        """
        if CV:
            reg = linear_model.RidgeCV(alphas=cv_range, cv=cv_fold)
        else:
            reg = linear_model.Ridge()
        return reg

    def lasso(self,
              CV=False, LAR=False, LARCV=False, LARIC=False,
              cv_range=None, cv_fold=None,
              criterion=None):
        """lasso regression, i.e., loss function + L1 regularization. this 
        regression provide two selection direction, model selection and feature 
        selection. model selection include CV & information criterion based 
        method, feature selection includes least angle regression.
        """
        if CV:
            reg = linear_model.LassoCV(alphas=cv_range, cv=cv_fold)
        elif LAR:
            reg = linear_model.LassoLars()
        elif LARCV:
            reg = linear_model.LassoLarsCV(cv=cv_fold)
        elif LARIC:
            reg = linear_model.LassoLarsIC(criterion=criterion)
        else:
            reg = linear_model.Lasso()
        return reg

    def elasticnet(self,
                   CV=False, l1_ratio=0.5, cv_fold=None):
        """
        Elastic-Net, i.e., loss function + L1 & L2 regularization. 

        NOTE: l1_ratio parmameters means the coefficient of L1 regularization, 
        the coefficient of L2 regularization is (1-l1_ratio).
        """
        if CV:
            reg = linear_model.ElasticNetCV(l1_ratio=l1_ratio, cv=cv_fold)
        else:
            reg = linear_model.ElasticNet()
        return reg

    def estimate(self, algorithm=None):
        """
        1. automatic relevance determination, 
        Compared to the OLS (ordinary least squares) estimator, the coefficient 
        weights are slightly shifted toward zeros, which stabilises them. 
        2. stochastic gradient descent method
        NOTE: could used for different penalty
        3. online passive-aggressive
        They are similar to the Perceptron in that they do not require a 
        learning rate. However, contrary to the Perceptron, they include a 
        regularization parameter C.
        """
        if algorithm == 1:
            reg = linear_model.ARDRegression()
        elif algorithm == 2:
            reg = linear_model.SGDRegressor()
        elif algorithm == 3:
            reg = linear_model.PassiveAggressiveRegressor()
        return reg

    def robust(self, algorithm=None):
        """
        Robust regression aims to fit a regression model in the presence of 
        corrupt data: either outliers, or error in the model.
        1. RANSAC
        2. Theil-Sen
        3. Huber
        """
        if algorithm == 1:
            reg = linear_model.RANSACRegressor()
        elif algorithm == 2:
            reg = linear_model.TheilSenRegressor()
        elif algorithm == 3:
            reg = linear_model.HuberRegressor()
        return reg


class Support_Vector_Regression():

    def __init__(self):
        pass

    def svr(self):
        reg = svm.SVR()
        return reg

    def nusvr(self):
        reg = svm.NuSVR()
        return reg

    def gridsearch(self, reg, tuned_params, score='neg_mean_squared_error'):
        reg = GridSearchCV(reg, tuned_params, scoring=score)
        return reg


class Tree_Regression():

    def __init__(self):
        pass

    def decision_tree(self):
        reg = tree.DecisionTreeRegressor()
        return reg

    def bagging(self, algorithm=None):
        if algorithm == 1:
            reg = ensemble.RandomForestRegressor()
        elif algorithm == 2:
            reg = ensemble.ExtraTreesRegressor()
        return reg

    def boosting(self, algorithm=None):
        if algorithm == 1:
            reg = ensemble.AdaBoostRegressor()
        elif algorithm == 2:
            reg = ensemble.GradientBoostingRegressor()
        elif algorithm == 3:
            reg = xgboost.XGBRegressor(n_estimators=500)
        elif algorithm == 4:
            reg = lightgbm.LGBMRegressor(n_estimators=100)
        return reg

    def gridsearch(self, reg, tuned_params, score='neg_mean_squared_error'):
        reg = GridSearchCV(reg, tuned_params, scoring=score)
        return reg


class Nerual_Network_Regression():

    def __init__(self):
        pass

    def ANN(self):
        reg = neural_network.MLPRegressor()
        return reg


class Gaussian_Process_Regression():

    def __init__(self):
        pass

    def GPR(self):
        reg = gaussian_process.GaussianProcessRegressor()
        return reg


class ELM():
    pass
