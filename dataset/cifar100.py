from torchvision import transforms
from torchvision.datasets import CIFAR100

from dataset.utils import get_dataset


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
    return get_dataset(dataset, train, transform_test, transform_train)
