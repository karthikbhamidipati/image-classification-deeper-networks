from os.path import basename, abspath, dirname

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

from dataset.cifar10 import cifar10
from dataset.cifar100 import cifar100
from dataset.mnist import mnist

PROJECT_NAME = basename(dirname(dirname(abspath(__file__))))

DATA_SOURCES = {
    "mnist": mnist,
    "cifar10": cifar10,
    "cifar100": cifar100
}

NETWORKS = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152
}

HYPER_PARAMETERS = {
    'num_epochs': 50,  # number of epochs
    'batch_size': 256,  # batch size
    'learning_rate': 0.001,  # learning rate
}
