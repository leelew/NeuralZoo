from MetReg.benchmark.metrics import (RMSE, Bias, InterannualVariablity,
                                      SpatialDist, CriterionScore, PhaseShift,
                                      RegressionScore)


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
                 ):
        pass

    def cal_overall_score(self):
        pass

    def cal_overall_score_1d(self):
        pass

    def _benchmark_array(self, y_true, y_pred):
        """time series score.

        Args:
            y_true ([type]): shape of (timesteps,)
            y_pred ([type]): shape of (timesteps,)
        """
        pass
