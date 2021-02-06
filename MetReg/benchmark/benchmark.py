from MetReg.benchmark.metrics import rmse, nse, mae, r2


class ScoreBoard():
    """A class for score board for spatial & temporal analysis.
    """
    
    def __init__(self, 
                 y_true, 
                 y_pred,
                 benchmark_mode=None):
        self.benchmark_mode = benchmark_mode
        if self.benchmark_mode is None:
            assert y_true.ndims == 3
        elif self.benchmark_mode == 'temporal':
            assert y_true.ndims 

    def _set_benchmark_mode(self, X):
        if X.ndims == 3:
            self.benchmark_mode = 'whole'
        elif X.ndims == 2:
            self.benchmark_mode = 'img'
        elif X.ndims == 1:
            self.benchmark_mode = 'array'

    def score(self):
        pass    

    def _benchmark_array(self, y_true, y_pred):
        """time series score.

        Args:
            y_true ([type]): shape of (timesteps,)
            y_pred ([type]): shape of (timesteps,)
        """
        
        
    def _benchmark_img(self, y_true, y_pred):
        pass

    def _benchmark_whole(self, y_true, y_pred):
        pass

    
