from MetReg.models.ml.linear import LR, ElasticNet, Lasso, Ridge
from MetReg.models.ml.tree import DT, GBDT, RF, LightGBM, Xgboost
from MetReg.utils.parser import get_lr_args

from MetReg.train.train_ml import train_ml
from MetReg.benchmark import _benchmark_array, _benchmark_img


class model_generator:
    """generate model according model name."""

    def __init__(self, mdl_name):
        self.mdl_name = mdl_name

        self.mdl_type = mdl_name.split('.')[0]

    def __call__(self):
        if 'lr' in self.mdl_name.lower():
            mdl = self._get_lr_mdl(self.mdl_name)
        elif 'tree' in self.mdl_name.lower():
            mdl = self._get_tree_mdl(self.mdl_name)
        else:
            raise NameError('Have not support this model!')
        return mdl

    def _get_lr_mdl(self, mdl_name):
        config = get_lr_args()

        lr_hash = {
            'lr': LR(config)(),
            'ridge': Ridge(config)(),
            'lasso': Lasso(config)(),
            'elasticnet': ElasticNet(config)(),
        }
        return lr_hash[mdl_name.split('.')[-1]]

    def _get_tree_mdl(self, mdl_name):

        lr_hash = {
            'dt': DT()(),
            'rf': RF()(),
            'gbdt': GBDT()(),
            'xgboost': Xgboost()(),
            'lightgbm': LightGBM()(),
        }
        return lr_hash[mdl_name.split('.')[-1]]


class model_trainer:

    def __init__(self,
                 mdl,
                 X,
                 y=None,
                 mdl_type='ml'):
        self.mdl = mdl
        self.X = X
        self.y = y
        self.mdl_type = mdl_type

    def __call__(self):
        if self.mdl_type == 'ml':
            self.mdl = train_ml(self.mdl, self.X, self.y)
        return self.mdl


class model_benchmarker:

    def __init__(self,
                 mdl,
                 X,
                 y=None):
        self.mdl = mdl
        self.X = X
        self.y = y

    def __call__(self):
        y_pred = self.mdl.predict(self.X)
        return _benchmark_array(self.X, y_pred)()


class model_loader:
    pass


class model_saver:
    pass
