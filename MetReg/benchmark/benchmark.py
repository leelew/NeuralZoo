from MetReg.benchmark.metrics import (RMSE, Bias, InterannualVariablity,
                                      SpatialDist, CriterionScore, PhaseShift,
                                      RegressionScore, OverallScore)


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
                 y_true,
                 y_pred,
                 score_list=None,
                 mode=None):
        if mode is None:
            self._get_benchmark_mode()
        else:
            self.mode = mode

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
        return OverallScore().score_1d(y_true, y_pred)

    def _benchmark_image_pools(self, y_true, y_pred):
        return OverallScore().score_3d(y_true, y_pred)
