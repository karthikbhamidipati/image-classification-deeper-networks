from torch.nn import Conv2d, Linear, DataParallel
from torchvision import models


def _get_vgg_model(vgg_model, input_filters, num_classes):
    conv1 = vgg_model.features[0]
    fc = vgg_model.classifier[-1]
    vgg_model.features[0] = Conv2d(input_filters, conv1.out_channels, kernel_size=conv1.kernel_size,
                                   stride=conv1.stride, padding=conv1.padding, bias=False)
    vgg_model.classifier[-1] = Linear(in_features=fc.in_features, out_features=num_classes,
                                      bias=True)
    return DataParallel(vgg_model)


def vgg11(input_filters, num_classes):
    return _get_vgg_model(models.vgg11(), input_filters, num_classes)


def vgg13(input_filters, num_classes):
    return _get_vgg_model(models.vgg13(), input_filters, num_classes)


def vgg16(input_filters, num_classes):
    return _get_vgg_model(models.vgg16(), input_filters, num_classes)


def vgg19(input_filters, num_classes):
    return _get_vgg_model(models.vgg19(), input_filters, num_classes)
