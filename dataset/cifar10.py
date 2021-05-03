from torchvision import transforms
from torchvision.datasets import CIFAR10


def cifar10(root_dir, train):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
        ]
    )

    return CIFAR10(root=root_dir, train=train, download=True, transform=transform)
