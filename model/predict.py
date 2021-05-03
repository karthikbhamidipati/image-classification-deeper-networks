from math import ceil

import torch
import wandb

from model import run_device
from model.metrics import compute_metrics


def predict(model, data_loader, criterion):
    num_iters = ceil(len(data_loader.dataset) / data_loader.batch_size)
    # TODO update initialization
    loss, metrics = 0, 0

    model.eval()
    with torch.no_grad():
        for data, labels in data_loader:
            data, label = data.to(run_device), labels.to(run_device)
            output = model(data)
            loss += criterion(output, label)
            metrics += compute_metrics(output, label)

    return loss / num_iters, metrics * 100 / num_iters


def log_pred_metrics(loss, metrics):
    print("Test loss: {}, Test Metrics: {}".format(loss, metrics))
    wandb.log({'test_loss': loss}, sync=False)
    wandb.log({'test_metrics': metrics})
