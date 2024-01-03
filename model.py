from torch import nn
import torch
import torch.nn.functional as F

class MyAwesomeModel(nn.Module):
    """My awesome model."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.maxpool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(3, 64, 3)

        # Dynamic computation of the size after max-pooling
        self.pool_output_size = self.calculate_pool_output_size()

        self.fc1 = nn.Linear(self.pool_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

    def calculate_pool_output_size(self):
        # Dummy input to compute the size after max-pooling
        x = torch.randn(1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        out_size = x.view(1, -1).size(1)
        return out_size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        # Flatten input to fit maxpool output
        x = x.view(-1, self.pool_output_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)

        return x


# Test output dimensions
if __name__ == "__main__":
    from data import mnist
    trainset, testset = mnist()
    model = MyAwesomeModel()
    # Make one forward pass
    images, labels = next(iter(trainset))
    out = model(images)
    print(out.shape)


