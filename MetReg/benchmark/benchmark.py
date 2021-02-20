import numpy as np
from MetReg.benchmark.metrics import (RMSE, Bias, CriterionScore,
                                      DefaultRegressionScore,
                                      InterannualVariablity, OverallScore,
                                      PhaseShift, SpatialDist)


class ScoreBoard():
    """A class for score board for spatial & temporal analysis using
    ILAMB scores and calculation methods.

    Notes:: This class now only support 3d input dataset shape of (timestep,
            lat, lon). We also provide 1d input dataset shape of (timestep,)
            which provide api for users need to benchmark on site performance.
            1d scoreboard only support mean state analysis on temporal 
            dimensions and also aware of `SpatialDist` which need more than 
            3d input datasets.
    """

    def __init__(self,
                 mode=None,
                 score_list=None,
                 overall_score=False):
        if mode is None:
            self._get_benchmark_mode()
        else:
            self.mode = mode
        
        self.overall_score=overall_score

    def benchmark(self, y_true, y_pred):
        if self.mode == 1:
            return self._benchmark_array(y_true, y_pred)
        else:
            return self._benchmark_image_pools(y_true, y_pred)

    def _get_benchmark_mode(self):
        pass


    def _benchmark_array(self, y_true, y_pred):
        """time series score.

        Args:
            y_true ([type]): shape of (timesteps,)
            y_pred ([type]): shape of (timesteps,)
        """

        if self.overall_score:
            return OverallScore().score_1d(y_true, y_pred)
        else:
            return DefaultRegressionScore().cal(y_true, y_pred)

    def _benchmark_image_pools(self, y_true, y_pred):

        if self.overall_score:
            return OverallScore().score_3d(y_true, y_pred)
        else:
            drs = DefaultRegressionScore()
            _, Nlat, Nlon = y_true.shape
            score_ = np.full((12, Nlat, Nlon), np.nan)

            for i in range(Nlat):
                for j in range(Nlon):
                    if not np.isnan(y_pred[:,i,j]).any():
                        print(y_pred[:,i,j])
                        score = drs.cal(y_true[:,i,j], y_pred[:,i,j])
                        score_[:, i, j] = np.array(score)
            return score_

    
