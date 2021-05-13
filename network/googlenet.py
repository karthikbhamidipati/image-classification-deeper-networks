from torch.nn import Conv2d, Linear
from torchvision import models


def _get_googlenet_model(googlenet_model, input_filters, num_classes):
    conv = googlenet_model.conv1.conv
    googlenet_model.conv1.conv = Conv2d(input_filters, conv.out_channels, kernel_size=conv.kernel_size,
                                        stride=conv.stride, padding=conv.padding, bias=False)
    googlenet_model.aux1.fc2 = Linear(in_features=1024, out_features=num_classes,
                                      bias=True)
    googlenet_model.aux2.fc2 = Linear(in_features=1024, out_features=num_classes,
                                      bias=True)
    googlenet_model.fc = Linear(in_features=1024, out_features=num_classes,
                                bias=True)
    return googlenet_model


def googlenet(input_filters, num_classes):
    return _get_googlenet_model(models.googlenet(), input_filters, num_classes)
