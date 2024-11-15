import pandas as pd
from matplotlib import pyplot as plt


def smoothed_plot(values, name):

    """Gives smoothed plot.

    Args:
        values: values to plot
        name: plot's title
    """

    plt.figure()
    pd.Series(values).rolling(100).mean().plot()
    plt.title(name)
    plt.show()
