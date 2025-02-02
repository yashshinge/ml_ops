"""Tests for the module model.py"""

# pylint: disable=C0103,W0621

import os
import sys
import pytest

import torch
from torch import nn
from model import SimpleClassifier

sys.path.insert(1, os.path.join(sys.path[0], '..'))

# Define inputs to the model
num_classes = 10
input_size = 784
num_hidden_units = 12


@pytest.fixture(scope='module')
def init_sc():
    """Setup method for initializing the SimpleClassifier class."""

    simple_classifier_model = SimpleClassifier(num_classes=num_classes,
                                               input_size=input_size,
                                               num_hidden_units=num_hidden_units)
    return simple_classifier_model


def test_init(init_sc):
    """Test model initialization and architecture."""

    assert isinstance(init_sc.model[0], nn.modules.linear.Linear)
    assert init_sc.model[0].weight.shape == (num_hidden_units, input_size)

    assert isinstance(init_sc.model[1], nn.modules.activation.LeakyReLU)

    assert isinstance(init_sc.model[2], nn.modules.linear.Linear)
    assert init_sc.model[2].weight.shape == (num_classes, num_hidden_units)

    assert isinstance(init_sc.model[3], nn.modules.activation.LogSoftmax)


def test_forward(init_sc):
    """Test forward pass functionality."""

    # pylint: disable=E1101
    inp_img = torch.randn(num_hidden_units, input_size)
    out = init_sc.forward(in_image=inp_img)
    assert out.shape == (len(inp_img), num_classes)
