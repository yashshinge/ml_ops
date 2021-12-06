"""Test train"""
# pylint: disable=C0103,E1121

import os
import sys
import io
from contextlib import redirect_stdout

import torch
from torch import optim
from torchvision import datasets, transforms

import train
from model import SimpleClassifier

sys.path.insert(1, os.path.join(sys.path[0], '..'))


def get_fake_data(is_train):
    """get data"""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    if is_train:
        return datasets.FakeData(size=1000, transform=transform)
    return datasets.FakeData(size=100, transform=transform)


def get_params(is_train):
    """get params"""
    dataset = get_fake_data(is_train)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    model = SimpleClassifier(num_classes=10, input_size=224 * 224 * 3)

    if is_train:
        optimizer = optim.Adadelta(model.parameters(), lr=0.01)
        epoch = 1
        return model, data_loader, optimizer, epoch
    return model, data_loader


def test_train():
    """test tra"""
    f = io.StringIO()
    with redirect_stdout(f):
        train.train(*get_params(is_train=True))
    out = f.getvalue()
    assert 'Train Epoch' in out


def test_test():
    """test test"""
    test_score = train.test(*get_params(is_train=False))

    assert isinstance(test_score, float)
    assert test_score >= 0.0
    assert test_score <= 100.0
