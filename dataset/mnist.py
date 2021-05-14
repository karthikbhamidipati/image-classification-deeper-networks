import urllib

from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from dataset.wrapper import DatasetWrapper


def mnist(root_dir, train):
    # Normalize the training set with augmentation
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize(0.406, 0.225, inplace=True)
        ]
    )

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize(0.406, 0.225, inplace=True)
        ]
    )

    opener = urllib.request.URLopener()
    opener.addheader('User-Agent',
                     'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 '
                     '(KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36')

    dataset = MNIST(root=root_dir, train=train, download=True)
    num_classes = dataset.classes
    if train:
        train_set, val_set = random_split(dataset, (int(0.8 * len(dataset)), int(0.2 * len(dataset))))
        return DatasetWrapper(train_set, num_classes, transform_train), DatasetWrapper(val_set, num_classes, transform_test)
    else:
        return DatasetWrapper(dataset, transform_test)
