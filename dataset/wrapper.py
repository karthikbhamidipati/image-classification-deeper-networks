from torch.utils.data import Dataset


class DatasetWrapper(Dataset):
    def __init__(self, dataset, num_classes, transform=None):
        self.dataset = dataset
        self.classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.transform(image), label