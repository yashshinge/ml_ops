""""""

import sys
import argparse
import logging

from uniplot import plot


def set_logger(level):
    """"""
    logger_ = logging.getLogger()
    logger_.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    logger_.addHandler(handler)
    return logger_


def get_args():
    """"""
    parser = argparse.ArgumentParser(description="Simple classifier training job.")
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help="Set log level (default: 'INFO')")
    args_ = vars(parser.parse_args())
    return args_


def plot_helper(x, y, num_epochs):
    """"""
    if (not x) or (not y) or not(num_epochs):
        print('No plot to display. Please check if the model training is run successfully.')
        return

    plot(xs=x, ys=y, x_min=1, x_max=num_epochs + 1, y_min=max(min(y) - 2, 0), y_max=min(max(y) + 2, 100),
         y_unit="%", y_gridlines=y,
         lines=True, title="Plot for test accuracy (y) v/s epochs (x)")
