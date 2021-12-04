""""""

import sys
import argparse
import logging

from uniplot import plot


def set_logger(level):
    """"""
    logger_ = logging.getLogger(__name__)
    logger_.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    logger_.addHandler(handler)
    return logger_


def get_args():
    """"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log_level', default='INFO', type=str, metavar='', help='Set log level')
    args_ = vars(parser.parse_args())
    return args_


def plot_helper(x, y, num_epochs):
    if (not x) or (not y) or not(num_epochs):
        print('No plot to display. Check if model training in complete')
        return

    plot(xs=x, ys=y, x_min=1, x_max=num_epochs + 1,
        y_min=max(min(y) - 2, 0), y_max=min(max(y) + 2, 100),
         lines=True, title="Plot for test accuracy (y) v/s epochs (x)")
