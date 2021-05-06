import urllib

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

    opener = urllib.request.URLopener()
    opener.addheader('User-Agent',
                     'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 '
                     '(KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36')

    return MNIST(root=root_dir, train=train, download=True, transform=transform)
