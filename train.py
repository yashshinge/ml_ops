"""Train simple model.

Inspired by the Pytorch example: https://github.com/pytorch/examples/blob/master/mnist/main.py
"""


import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from model import SimpleClassifier
from utils import get_args, set_logger, plot_helper


def train(model, train_loader, optimizer, epoch):
    """"""
    model.train()
    log_interval = 1000
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            logger.debug(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                         f"({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")


def test(model, test_loader):
    """"""
    model.eval()
    test_loss = 0
    correct = 0
    len_test_loader = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len_test_loader
    accuracy = 100.0 * correct / len_test_loader

    logger.debug(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len_test_loader} ({accuracy:.0f}%)\n")

    return accuracy


if __name__ == "__main__":

    args = get_args()
    logger = set_logger(args.log_level)

    torch.manual_seed(args.seed)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset_train = datasets.MNIST(
        "data", train=True, download=True, transform=transform
    )
    dataset_test = datasets.MNIST("data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=10)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=10)
    model = SimpleClassifier(num_classes=10, input_size=28 * 28)
    optimizer = optim.Adadelta(model.parameters(), lr=0.01)

    # Train and validate.
    num_epochs = 2
    training_start_time = time.perf_counter()

    test_accuracy = {k: None for k in range(1, num_epochs + 1)}

    for epoch in range(1, num_epochs + 1):
        train(model, train_loader, optimizer, epoch)
        test_accuracy[epoch] = test(model, test_loader)

    total_training_time = time.perf_counter() - training_start_time
    logger.info(f'Total training time: {total_training_time:.3f} secs')
    logger.info(f'Final test accuracy: {test_accuracy[num_epochs]:.4f}%\n')

    # Plotting graph for test accuracy v/s epochs
    x, y = list(test_accuracy), list(test_accuracy.values())
    plot_helper(x=x, y=y, num_epochs=num_epochs)
