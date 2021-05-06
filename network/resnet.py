from torch.nn import Conv2d, Linear
from torchvision import models


def _get_resnet_model(resnet_model, input_filters, num_classes):
    conv1 = resnet_model.conv1
    fc = resnet_model.fc
    resnet_model.conv1 = Conv2d(input_filters, conv1.out_channels, kernel_size=conv1.kernel_size,
                                stride=conv1.stride, padding=conv1.padding, bias=False)
    resnet_model.fc = Linear(in_features=fc.in_features, out_features=num_classes,
                             bias=True)
    return resnet_model


def resnet18(input_filters, num_classes):
    return _get_resnet_model(models.resnet18(), input_filters, num_classes)


def resnet34(input_filters, num_classes):
    return _get_resnet_model(models.resnet34(), input_filters, num_classes)


def resnet50(input_filters, num_classes):
    return _get_resnet_model(models.resnet50(), input_filters, num_classes)


def resnet101(input_filters, num_classes):
    return _get_resnet_model(models.resnet101(), input_filters, num_classes)


def resnet152(input_filters, num_classes):
    return _get_resnet_model(models.resnet152(), input_filters, num_classes)
