# dataloader.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from config import cfg
import numpy as np

def get_dataloaders():
    # 1. Augmentation 설정
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

    # 2. 전체 학습 데이터셋 로드 (인덱스 분리를 위해 Raw 데이터 먼저 준비)
    # transform은 나중에 Subset에 적용하는 것이 까다로우므로, 
    # 두 개의 데이터셋 객체를 만들고 인덱스로 나눕니다.
    full_trainset_aug = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    full_trainset_clean = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_test)

    # 3. Train / Val Split (45k / 5k)
    num_train = len(full_trainset_aug)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train)) # 10% for validation
    
    # Reproducibility를 위해 shuffle 후 split
    np.random.seed(cfg.seed)
    np.random.shuffle(indices)
    
    train_idx, val_idx = indices[split:], indices[:split]

    # Subset 생성 (Train엔 Augmentation, Val엔 Clean Transform 적용)
    train_ds = Subset(full_trainset_aug, train_idx)
    val_ds = Subset(full_trainset_clean, val_idx)
    
    # Test Dataset
    test_ds = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    # 4. DataLoader 생성
    trainloader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    print(f"Data Split -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    return trainloader, valloader, testloader