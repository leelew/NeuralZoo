import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.floating_axes as fa
import mpl_toolkits.axisartist.grid_finder as gf
import numpy as np
from matplotlib.projections import PolarAxes
from mpl_toolkits import Basemap


def plot_taylor_diagram(stdev,
                        corrcoeff,
                        refstd,
                        fig,
                        colors,
                        normalize=True):
    """Plot a taylor diagram

    Args:
        stddev : numpy.ndarray
            an array of standard deviations
        corrcoeff : numpy.ndarray
            an array of correlation coefficients
        refstd : float
            the reference standard deviation
        fig : matplotlib figure
            the matplotlib figure
        colors : array
            an array of colors for each element of the input arrays
        normalize : bool, optional
            disable to skip normalization of the standard deviation
    """
    # define transform
    tr = PolarAxes.PolarTransform()

    # correlation labels
    rlocs = np.concatenate((np.arange(10)/10., [0.95, 0.99]))
    tlocs = np.arccos(rlocs)
    gl1 = GF.FixedLocator(tlocs)
    tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

    # standard deviation axis extent
    if normalize:
        stddev = stddev/refstd
        refstd = 1.
    smin = 0
    smax = max(2.0, 1.1*stddev.max())

    # add the curvilinear grid
    ghelper = FA.GridHelperCurveLinear(tr,
                                       extremes=(0, np.pi/2, smin, smax),
                                       grid_locator1=gl1,
                                       tick_formatter1=tf1)
    ax = FA.FloatingSubplot(fig, 111, grid_helper=ghelper)
    fig.add_subplot(ax)

    # adjust axes
    ax.axis["top"].set_axis_direction("bottom")
    ax.axis["top"].toggle(ticklabels=True, label=True)
    ax.axis["top"].major_ticklabels.set_axis_direction("top")
    ax.axis["top"].label.set_axis_direction("top")
    ax.axis["top"].label.set_text("Correlation")
    ax.axis["left"].set_axis_direction("bottom")
    if normalize:
        ax.axis["left"].label.set_text("Normalized standard deviation")
    else:
        ax.axis["left"].label.set_text("Standard deviation")
    ax.axis["right"].set_axis_direction("top")
    ax.axis["right"].toggle(ticklabels=True)
    ax.axis["right"].major_ticklabels.set_axis_direction("left")
    ax.axis["bottom"].set_visible(False)
    ax.grid(True)

    ax = ax.get_aux_axes(tr)
    # Plot data
    corrcoef = corrcoef.clip(-1, 1)
    for i in range(len(corrcoef)):
        ax.plot(np.arccos(corrcoef[i]), stddev[i],
                'o', color=colors[i], mew=0, ms=8)

    # Add reference point and stddev contour
    l, = ax.plot([0], refstd, 'k*', ms=12, mew=0)
    t = np.linspace(0, np.pi/2)
    r = np.zeros_like(t) + refstd
    ax.plot(t, r, 'k--')

    # centralized rms contours
    rs, ts = np.meshgrid(np.linspace(smin, smax),
                         np.linspace(0, np.pi/2))
    rms = np.sqrt(refstd**2 + rs**2 - 2*refstd*rs*np.cos(ts))
    contours = ax.contour(ts, rs, rms, 5, colors='k', alpha=0.4)
    ax.clabel(contours, fmt='%1.1f')

    return ax


def _get_grid_attribute():
    pass


def plot_countourf(
        inputs,
        lat,
        lon,
        output_path):
    """plot countourf using basemap according lat, lon."""
    plt.figure()

    m = Basemap()
    m.drawcoastlines(linewidth=0.2)
    x, y = m(lon, lat)

    sc = m.pcolormesh(x, y, inputs)

    sc.set_edgecolor('face')

    m.colorbar(sc, location='bottom')
    plt.savefig(output_path)
    # plt.text()


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
