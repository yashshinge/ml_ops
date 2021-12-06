"""utils"""

import argparse

import matplotlib.pyplot as plt


def get_args():
    """args"""
    parser = argparse.ArgumentParser(description="Simple classifier training job.")
    parser.add_argument('--epochs', type=int, default=2, metavar='E',
                        help="Number of epochs to train (default: 2)")
    parser.add_argument('--seed', type=int, default=918, metavar='S',
                        help="Random seed (default: 918)")
    args = parser.parse_args()
    return args


def plot_helper(x, y, plt_name):
    """plot"""

    # pylint: disable=C0103  # To maintain matplotlib standards of variable naming.

    if (not isinstance(x, list)) or (not isinstance(y, list)):
        raise TypeError("Function expects type 'list' for inputs x and y.")

    plt.style.use('seaborn-darkgrid')
    ax = plt.axes()
    ax.set(xticks=x, xlabel="No. of epochs", ylabel='Accuracy (%)', title='Test accuracy v/s Epochs')
    _ = plt.plot(x, y)
    plt.savefig(plt_name)
