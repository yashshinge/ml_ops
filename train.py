"""Train simple model.

Inspired by the Pytorch example: https://github.com/pytorch/examples/blob/master/mnist/main.py
"""
import torch
from torchvision import datasets, transforms
from model import SimpleClassifier
import torch.optim as optim
import torch.nn.functional as F


def train(model, train_loader, optimizer, epoch):
    model.train()
    log_interval = 1000
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


if __name__ == "__main__":
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
    for epoch in range(1, num_epochs + 1):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)
