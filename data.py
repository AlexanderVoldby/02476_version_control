import torch
from torchvision import datasets, transforms

def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    train = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


    testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
    test = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    return train, test
