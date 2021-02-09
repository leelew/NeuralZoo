import six
import abc

import numpy as np


@six.add_metaclass(abc.ABCMeta)
class BaseScore():

    def __init__(self):
        pass

    def score(self,
              func,
              y_true,
              y_pred,
              mass_weight=True):
        """Mean values over space of bias score.
        Args:
            y_true, y_pred(nd.array):
                shape of(timesteps, height, width)
        """
        T, Nlat, Nlon = y_true.shape
        score_ = [func(y_true[:, i, j], y_pred[:, i, j])
                  for i in range(Nlat) for j in range(Nlon)]

        if mass_weight:
            # mass weighting
            weight = np.mean(y_true, axis=0)
            score_ = np.mean(np.multiarray(score_, weight))
        return score_

    def __repr__(self):
        pass

    @staticmethod
    def _print_avg_metrics(metrics):

        for metric_name, metric in metrics.items():
            print('{} is {}'.format(metric_name, np.nanmean(metric)))
