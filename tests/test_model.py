"""TEST MODEL """

# pylint: disable=C0103,W0621

import os
import sys
import pytest

import torch
import torch.nn as nn
from model import SimpleClassifier

sys.path.insert(1, os.path.join(sys.path[0], '..'))

# Define inputs to the model
num_classes = 10
input_size = 784
num_hidden_units = 10


@pytest.fixture(scope='module')
def sc():
    """sc setup"""

    simple_classifier_model = SimpleClassifier(num_classes=num_classes,
                                               input_size=input_size,
                                               num_hidden_units=num_hidden_units)
    return simple_classifier_model


def test_init(sc):
    """test init"""
    assert isinstance(sc.model[0], nn.modules.linear.Linear)
    assert sc.model[0].weight.shape == (num_hidden_units, input_size)

    assert isinstance(sc.model[1], nn.modules.activation.LeakyReLU)

    assert isinstance(sc.model[2], nn.modules.linear.Linear)
    assert sc.model[2].weight.shape == (num_classes, num_hidden_units)

    assert isinstance(sc.model[3], nn.modules.activation.LogSoftmax)


def test_forward(sc):
    """test fwd"""
    inp_img = torch.randn(num_hidden_units, input_size)
    out = sc.forward(in_image=inp_img)
    assert out.shape == (len(inp_img), num_classes)
