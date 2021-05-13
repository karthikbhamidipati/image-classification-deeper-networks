from os.path import basename, abspath, dirname

from dataset.cifar10 import cifar10
from dataset.cifar100 import cifar100
from dataset.mnist import mnist
from network.googlenet import googlenet
from network.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from network.vgg import vgg11, vgg13, vgg16, vgg19

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
    "resnet152": resnet152,
    "vgg11": vgg11,
    "vgg13": vgg13,
    "vgg16": vgg16,
    "vgg19": vgg19,
    "googlenet": googlenet
}

HYPER_PARAMETERS = {
    'num_epochs': 50,  # number of epochs
    'batch_size': 512,  # batch size
    'learning_rate': 0.01,  # learning rate
    'momentum': 0.9,  # Momentum of optimizer
    'min_learning_rate': 0.001  # minimum learning rate
}
