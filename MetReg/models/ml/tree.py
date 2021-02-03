
try:
    import lightgbm
    import xgboost
except:
    raise KeyError('run `pip3 install lightgbm xgboost`')

from MetReg.base.base_model import BaseModel
from sklearn import ensemble, tree
import numpy as np
np.random.seed(1)


class BaseTreeRegressor(BaseModel):
    """implementation of base decision tree regression.

    Args:
    criterion (str, optional):
        could be {"mse", "friedman_mse", "mae", "poisson"}, Defaults to 'mse'.
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion and minimizes the L2 loss
        using the mean of each terminal node, "friedman_mse", which uses mean
        squared error with Friedman's improvement score for potential splits,
        "mae" for the mean absolute error, which minimizes the L1 loss using
        the median of each terminal node, and "poisson" which uses reduction in
        Poisson deviance to find splits.
    splitter (str, optional):
        could be {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.
    max_depth (int, optional):
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples. Defaults to None.
    min_sample_split (int or float, optional):
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split. Defaults to 2.
    min_sample_leaf (int, optional):
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node. Defaults to 1.
    min_weight_fraction_leaf (float, optional):
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided. Defaults to 0.0.
    max_features ([type], optional):
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`. Defaults to None.

    Attributes:

    feature_importances_ : ndarray of shape (n_features,)
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.
    """

    def __init__(self,
                 criterion='mse',
                 splitter='best',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features=None,
                 ):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features

        self.regressor = None

    def fit(self, X, y):
        self.regressor = tree.DecisionTreeRegressor(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
        )
        self.regressor.fit(X, y)
        return self


class RandomForestRegressor(BaseTreeRegressor, BaseModel):
    """Random forest regressor."""

    def __init__(self,
                 n_estimators=100,
                 criterion='mse',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 n_jobs=-1,
                 verbose=0):
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
        )
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.regressor = None

    def fit(self, X, y):
        self.regressor = ensemble.RandomForestRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,  # care memory, default set as using all CPU.
            verbose=self.verbose,
        )
        self.regressor.fit(X, y)
        return self

    @staticmethod
    def get_hyperparameter_search_space():
        pass

    def __repr__(self):
        return {'short name': 'RF',
                'name': 'Random Forest'}


class ExtraTreesRegressor(RandomForestRegressor, BaseModel):

    def __init__(self, n_jobs=-1):
        self.regressor = None
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.regressor = ensemble.ExtraTreesRegressor(
            n_jobs=self.n_jobs,)
        self.regressor.fit(X, y)

        return self


class DeepForestRegressor(RandomForestRegressor, BaseModel):

    def __init__(self):
        self.regressor = None

    def fit(self, X, y):
        pass


class AdaptiveBoostingRegressor(RandomForestRegressor, BaseModel):
    def __init__(self, ):
        self.regressor = None

    def fit(self, X, y):
        self.regressor = ensemble.AdaBoostRegressor()
        self.regressor.fit(X, y)

        return self


class GradientBoostingRegressor(RandomForestRegressor, BaseModel):

    def __init__(self,
                 loss='ls',
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 criterion='friedman_mse',
                 max_depth=3,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features=None,):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
        )

        self.loss = loss
        self.learning_rate = 0.1
        self.subsample = subsample

        self.regressor = None

    def fit(self, X, y):
        self.regressor = ensemble.GradientBoostingRegressor(
            loss=self.loss,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
        )

        self.regressor.fit(X, y)
        return self


class ExtremeGradientBoostingRegressor(GradientBoostingRegressor, BaseModel):

    def __init__(self,
                 n_estimators=100,
                 max_depth=6,
                 learning_rate=0.3,
                 tree_method='auto',
                 subsample=1,
                 min_child_weight=1,
                 colsample_bytree=1,
                 n_jobs=-1
                 ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            subsample=subsample,
            learning_rate=learning_rate,
        )

        self.tree_method = tree_method
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree
        self.n_jobs = n_jobs
        self.regressor = None

    def fit(self, X, y):
        self.regressor = xgboost.XGBRegressor(
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            tree_method=self.tree_method,
            min_child_weight=self.min_child_weight,
            colsample_bytree=self.colsample_bytree,
            n_jobs=self.n_jobs,
        )
        self.regressor.fit(X, y)
        return self


class LightGradientBoostingRegressor(ExtremeGradientBoostingRegressor, BaseModel):

    def __init__(self,
                 num_leaves=31,
                 max_depth=-1,
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 min_child_samples=20,
                 min_child_weight=1e-3,
                 feature_fraction=1.0,
                 bagging_fraction=1.0,
                 reg_alpha=0.0,
                 reg_lambda=0.0,
                 ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            subsample=subsample,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
        )

        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

        self.regressor = None

    def fit(self, X, y):
        self.regressor = lightgbm.LGBMRegressor(
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            subsample=self.subsample,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            min_child_weight=self.min_child_weight,
            feature_fraction=self.feature_fraction,
            bagging_fraction=self.bagging_fraction,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
        )

        self.regressor.fit(X, y)
        return self


class CatBoostingRegressor(BaseModel):
    pass


if __name__ == "__main__":

    pass
