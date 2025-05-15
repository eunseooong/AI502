import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(config, dataset_name):
    if dataset_name == 'CIFAR100':
        transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=4)
        return trainloader
    elif dataset_name == 'MNIST':
        config.image_size = 28
        config.num_channels = 1
        config.num_classes = 10
        config.conditional = True
        transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=4)
        return trainloader
    else:
        raise ValueError("Unknown dataset")
