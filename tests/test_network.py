import logging
import os
import unittest
from os.path import join, dirname, abspath

from parameterized import parameterized
from torch.utils.data import DataLoader
from torchvision.models import GoogLeNetOutputs

from model.config import NETWORKS, DATA_SOURCES


class TestNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root_dir = join(dirname(abspath(__file__)), "data")
        if not os.path.exists(cls.root_dir):
            os.mkdir(cls.root_dir)
        cls.datasets = dict()
        for name, dataset in DATA_SOURCES.items():
            logging.info("Downloading {} dataset before running tests".format(name))
            cls.datasets[name] = dataset(root_dir=cls.root_dir, train=False)

    @parameterized.expand(NETWORKS.keys())
    def test(self, name):
        for dataset_name, dataset in self.datasets.items():
            input_filters = dataset[0][0].shape[0]
            num_classes = len(dataset.classes)
            data, _ = next(iter(DataLoader(dataset, batch_size=4)))
            model = NETWORKS[name](input_filters, num_classes)
            outputs = model(data)
            if not isinstance(outputs, GoogLeNetOutputs):
                outputs = [outputs]
            for output in outputs:
                self.assertEqual(output.shape[0], 4,
                                 "Incorrect output batch size for model: {}, dataset: {}"
                                 .format(name, dataset_name))
                self.assertEqual(output.shape[1], num_classes,
                                 "Incorrect output batch size for model: {}, dataset: {}"
                                 .format(name, dataset_name))


if __name__ == '__main__':
    unittest.main()
