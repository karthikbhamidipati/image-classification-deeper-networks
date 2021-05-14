from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import CIFAR100

from dataset.wrapper import DatasetWrapper


def cifar100(root_dir, train):
    # Normalize the training set with augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = CIFAR100(root=root_dir, train=train, download=True)
    if train:
        train_set, val_set = random_split(dataset, (int(0.8 * len(dataset)), int(0.2 * len(dataset))))
        return DatasetWrapper(train_set, transform_train), DatasetWrapper(val_set, transform_test)
    else:
        return DatasetWrapper(dataset, transform_test)
