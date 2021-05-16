from torch.utils.data import Dataset, random_split


class DatasetWrapper(Dataset):
    def __init__(self, dataset, num_classes, transform):
        self.dataset = dataset
        self.classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.transform(image), label


def get_dataset(dataset, train, transform_test, transform_train):
    num_classes = dataset.classes
    if train:
        train_set, val_set = random_split(dataset, (int(0.8 * len(dataset)), int(0.2 * len(dataset))))
        return DatasetWrapper(train_set, num_classes, transform_train), DatasetWrapper(val_set, num_classes,
                                                                                       transform_test)
    else:
        return DatasetWrapper(dataset, num_classes, transform_test)
