import unittest

import torch

from model.metrics import Metrics


class TestMetrics(unittest.TestCase):
    def test_metrics_initialization(self):
        metrics = Metrics(2)
        self.assertEqual(metrics.asdict(), {'loss': 0.0, 'accuracy': 0.0,
                                            'precision': 0.0, 'recall': 0.0,
                                            'fscore': 0.0, 'top_k_accuracy': 0.0},
                         "Initial metrics mismatch")

    def test_metrics_update(self):
        metrics = Metrics(2)
        metrics.update(torch.tensor(2.999),
                       torch.Tensor([[0.5, 0.2, 0.2],
                                     [0.3, 0.4, 0.2],
                                     [0.2, 0.4, 0.3],
                                     [0.7, 0.2, 0.1]]),
                       torch.Tensor([0, 1, 2, 0]))
        self.assertEqual(metrics.asdict(), {'loss': 3.0, 'accuracy': 75.0,
                                            'precision': 83.33, 'recall': 66.67,
                                            'fscore': 55.56, 'top_k_accuracy': 100.0},
                         "Metrics mismatch first update")

        metrics.update(torch.tensor(2.0),
                       torch.Tensor([[0.5, 0.2, 0.2],
                                     [0.3, 0.4, 0.2],
                                     [0.2, 0.4, 0.3],
                                     [0.7, 0.2, 0.1]]),
                       torch.Tensor([0, 1, 0, 2]))
        self.assertEqual(metrics.asdict(), {'loss': 2.5, 'accuracy': 62.5,
                                            'precision': 75.0, 'recall': 58.34,
                                            'fscore': 47.22, 'top_k_accuracy': 75.0},
                         "Metrics mismatch second update")


if __name__ == '__main__':
    unittest.main()
