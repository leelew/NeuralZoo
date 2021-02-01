import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import Basemap

def plot_countourf(inputs, lat, lon, output_path):
    """plot countourf using basemap according lat, lon."""
    plt.figure()

    m = Basemap()
    m.drawcoastlines(linewidth=0.2)
    x, y = m(lon, lat)

    sc = m.pcolormesh(x, y, inputs)

    plt.colorbar()
    plt.savefig('1.pdf')
    
    
def plot_scatter():
    """plot scatter and fitting line."""
    pass


def plot_plot():
    """plot timeseries and show shadow as std."""
    pass



class Plotting():
    """plot figures over train process.

    A class plot images or json generating over training process, including
    training time cost, history plotting, training parameters, etc.
    """

    def __init__(self): pass

    def plot_temp_figure(self, y_predict, y_valid, metrics, importance):
        # -----------------------------------
        # generate temperate figure for HTML
        # -----------------------------------
        plt.figure(figsize=(10, 10))

        # plot scatter of predict & truth
        ax = plt.subplot(2, 2, 1)
        plt.scatter(y_predict, y_valid)

        ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls='--', c='red')
        plt.xlabel('model prediction')
        plt.ylabel('ground truth')
        plt.xlim(min(y_predict), max(y_predict))
        plt.ylim(min(y_predict), max(y_predict))

        a, b = np.polyfit(y_predict, y_valid, deg=1)
        y_est = a * y_predict + b
        ax.plot(y_predict, y_est, '-')
        plt.title('scatter of predict & truth')

        # plot importance (only for linear regression)
        if not isinstance(importance, bool):
            plt.subplot(2, 2, 2)
            plt.bar(range(len(importance)), importance)
            plt.axhline(y=0, c='red')
            plt.title('importance of linear regression')

        # plot time series of predict & truth
        plt.subplot(2, 1, 2)
        plt.plot(y_predict, c='hotpink')
        plt.plot(y_valid, c='lime')
        plt.legend(['model prediction', 'ground truth'])
        plt.title('time series of predict & truth')

        plt.savefig('1.png')
