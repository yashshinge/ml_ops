
import torch.nn as nn
class SimpleClassifier(nn.Module):
    """Simple classifier with one hidden layer."""

    def __init__(self, num_classes, input_size, num_hidden_units=10):
        super(SimpleClassifier, self).__init__()
         
        self.model = nn.Sequential(
            nn.Linear(input_size, num_hidden_units),
            nn.LeakyReLU(),
            nn.Linear(num_hidden_units, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, in_image):
        input_vector = in_image.view(in_image.shape[0], -1)
        return self.model(input_vector)
