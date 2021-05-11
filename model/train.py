import logging

import torch
import wandb
from numpy import Inf

from model import run_device
from model.metrics import Metrics
from model.predict import predict


def train_model(model, train_loader, criterion, optimizer):
    metrics = Metrics()

    model.train()
    for data, labels in train_loader:
        data, label = data.to(run_device), labels.to(run_device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        metrics.update(loss, output, label)

    return metrics.asdict()


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_path):
    # wandb.watch(model, log='all')
    val_loss_min = Inf

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_model(model, train_loader, criterion, optimizer)
        val_metrics = predict(model, val_loader, criterion)
        log_metrics(epoch, train_metrics, val_metrics)
        val_loss_min = save_model(model, model_path, val_metrics['loss'], val_loss_min)

    wandb.save(model_path)


def save_model(model, model_path, val_loss, val_loss_min):
    if val_loss <= val_loss_min:
        logging.info("Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(val_loss_min, val_loss))
        torch.save(model, model_path)

    return min(val_loss, val_loss_min)


def log_metrics(epoch, train_metrics, val_metrics):
    wandb.log({'train': train_metrics}, step=epoch)
    wandb.log({'val': val_metrics}, step=epoch)
    logging.info("Epoch: {}\n\t"
                 "Train stats: {}\n\t"
                 "Val stats: {}\n\t"
                 .format(epoch, train_metrics, val_metrics))
