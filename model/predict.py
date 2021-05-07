import logging

import torch
import wandb

from model import run_device
from model.metrics import Metrics


def predict(model, data_loader, criterion):
    metrics = Metrics()

    model.eval()
    with torch.no_grad():
        for data, labels in data_loader:
            data, label = data.to(run_device), labels.to(run_device)
            output = model(data)
            loss = criterion(output, label)
            metrics.update(loss, output, label)

    return metrics.asdict()


def log_pred_metrics(metrics):
    logging.info("Test stats: {}".format(metrics))
    wandb.log({'test': metrics})
