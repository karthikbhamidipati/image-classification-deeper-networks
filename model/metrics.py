import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, top_k_accuracy_score


class Metrics:
    def __init__(self, k=5):
        self._n = 1
        self._k = k
        self.loss = 0.0
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.fscore = 0.0
        self.top_k_accuracy = 0.0

    def update(self, loss, prediction, ground_truth):
        self.loss = self._compute_moving_average(self.loss, loss.cpu().item())
        self._update_metrics(prediction.cpu().numpy(), ground_truth.cpu().numpy())
        self._n += 1

    def _compute_moving_average(self, prev, curr):
        return round(prev + ((curr - prev) / self._n), 2)

    def asdict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def __getitem__(self, item):
        return getattr(self, item)

    def _update_metrics(self, prediction, ground_truth):
        top_k_accuracy = top_k_accuracy_score(ground_truth, prediction, k=self._k)
        prediction = prediction.argmax(axis=1)
        accuracy = accuracy_score(ground_truth, prediction)
        precision, recall, fscore, _ = precision_recall_fscore_support(ground_truth, prediction, average='macro',
                                                                       zero_division=1)

        self.accuracy = self._compute_moving_average(self.accuracy, accuracy * 100)
        self.top_k_accuracy = self._compute_moving_average(self.top_k_accuracy, top_k_accuracy * 100)
        self.precision = self._compute_moving_average(self.precision, precision * 100)
        self.recall = self._compute_moving_average(self.recall, recall * 100)
        self.fscore = self._compute_moving_average(self.fscore, fscore * 100)
