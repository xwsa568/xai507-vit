# dataloader.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from config import cfg
import numpy as np

def get_dataloaders():
    # 1. Define Augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 2. Load full training dataset (Prepare raw data first for index splitting)
    # Applying transform to Subset later can be tricky,
    # so we create two dataset objects and split them by index.
    full_trainset_aug = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    full_trainset_clean = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_test)

    # 3. Train / Val Split (45k / 5k)
    num_train = len(full_trainset_aug)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train)) # 10% for validation
    
    # Shuffle and split for reproducibility
    np.random.seed(cfg.seed)
    np.random.shuffle(indices)
    
    train_idx, val_idx = indices[split:], indices[:split]

    # Create Subsets (Apply Augmentation to Train, Clean Transform to Val)
    train_ds = Subset(full_trainset_aug, train_idx)
    val_ds = Subset(full_trainset_clean, val_idx)
    
    # Test Dataset
    test_ds = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    # 4. Create DataLoaders
    trainloader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    print(f"Data Split -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    return trainloader, valloader, testloader