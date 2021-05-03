from torchvision import transforms
from torchvision.datasets import MNIST


def mnist(root_dir, train):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize(0.406, 0.225, inplace=True)
        ]
    )

    return MNIST(root=root_dir, train=train, download=True, transform=transform)
