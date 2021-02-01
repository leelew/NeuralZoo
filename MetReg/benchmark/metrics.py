# -----------------------------------------------------------------------------
#                                Evaluate module                              # 
# -----------------------------------------------------------------------------
# author: Lu Li                                                               #
# mail: lilu35@mail2.sysu.edu.cn                                              #
# -----------------------------------------------------------------------------
# This Repo is for model performance based on all ILAMB metrics (https://doi. #
# org/10.1029/2018MS001354), including sklearn based metrics, unbiased metric #
# -----------------------------------------------------------------------------


from sklearn.metrics import (explained_variance_score, max_error,
                             mean_absolute_error, mean_gamma_deviance,
                             mean_poisson_deviance, mean_squared_error,
                             mean_squared_log_error, mean_tweedie_deviance,
                             median_absolute_error, r2_score)
import numpy as np

class Metrics():

    def __init__(self, validate, forecast, metrics:list=None):

        if metrics is None:
            self.metrics = True
        else:
            self.metrics = metrics

        self._get_data_attr(validate)

    def _get_data_attr(self, validate):
        self.T, self.H, self.W = validate.shape

    def get_sklearn_metrics(self, validate,forecast):

        """get regression metrics from sklearn API.

        Args:
            forecast (np.array):
                shape of (timestep, height, width)
            validate (np.array):
                shape of (timestep, height, width) 

        Returns:
            metrics (dict):
        """
        metrics = {}
        
        validate = validate.reshape(self.T, self.H*self.W)
        forecast = forecast.reshape(self.T, self.H*self.W)

        metrics['evs'] = explained_variance_score(validate, forecast, multioutput='raw_values').reshape(self.H*self.W)
        metrics['mae'] = mean_absolute_error(validate, forecast,multioutput='raw_values').reshape(self.H*self.W)
        metrics['mse'] = mean_squared_error(validate, forecast,multioutput='raw_values').reshape(self.H*self.W)
        #metrics['msle'] = mean_squared_log_error(validate, forecast,multioutput='raw_values').reshape(self.H*self.W)
        #metrics['me'] = max_error(validate, forecast,multioutput='raw_values').reshape(self.H*self.W)
        metrics['r2'] = r2_score(validate, forecast,multioutput='raw_values').reshape(self.H*self.W)
        #metrics['medae'] = median_absolute_error(validate, forecast,multioutput='raw_values').reshape(self.H*self.W)
        #metrics['mpd'] = mean_poisson_deviance(validate, forecast,multioutput='raw_values').reshape(self.H*self.W)
        #metrics['mgd'] = mean_gamma_deviance(validate,forecast,multioutput='raw_values').reshape(self.H*self.W)
        #metrics['mtd'] = mean_tweedie_deviance(validate, forecast,multioutput='raw_values').reshape(self.H*self.W)

        print(metrics['r2'].reshape(-1,))
        return metrics
    
    @staticmethod
    def _print_avg_metrics(metrics):

        for metric_name, metric in metrics.items():
            print('{} is {}'.format(metric_name, np.nanmean(metric)))


    @staticmethod
    def get_unbiased_metrics(forcast, validate):
        pass
    

    def get_criterion_metrics(self):
        """aic, aicc, bic, mallows cp
        """
        pass

    def get_importance_metrics(self):
        """
        """
        # generate importance array
        try:
            importance = reg.coef_ * x_train.std(axis=0)
        except AttributeError:  # for tree regression
            try:  # only for RANSAC, cuz different code of using coef_
                importance = reg.estimator_.coef_ * x_train.std(axis=0)
            except AttributeError:
                try:
                    importance = reg.feature_importances_
                except:
                    importance = False
                    print('This is not a linear regression or tree regression')

        return y_predict, importance