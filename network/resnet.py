import torchvision.models as models
from torch.nn import Conv2d, Linear


def _get_resnet_model(resnet_model, input_filters, num_classes):
    resnet_model.conv1 = Conv2d(input_filters, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet_model.fc = Linear(in_features=resnet_model.fc.in_features, out_features=num_classes, bias=True)
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
