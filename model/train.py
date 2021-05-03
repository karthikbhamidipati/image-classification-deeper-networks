from math import ceil

import torch
import wandb
from numpy import Inf

from model import run_device
from model.metrics import compute_metrics
from model.predict import predict


def train_model(model, train_loader, criterion, optimizer):
    num_iters = ceil(len(train_loader.dataset) / train_loader.batch_size)
    # TODO update initialization
    train_loss, train_metrics = 0, 0

    model.train()
    for data, labels in train_loader:
        data, label = data.to(run_device), labels.to(run_device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_metrics += compute_metrics(outputs, labels)

    return train_loss / num_iters, train_metrics * 100 / num_iters


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_path):
    wandb.watch(model, log='all')
    val_loss_min = Inf

    for epoch in range(num_epochs):
        train_loss, train_metrics = train_model(model, train_loader, criterion, optimizer)
        val_loss, val_metrics = predict(model, val_loader, criterion)
        log_metrics(epoch, train_loss, train_metrics, val_loss, val_metrics)
        val_loss_min = save_model(model, model_path, val_loss, val_loss_min)

    wandb.save(model_path)


def save_model(model, model_path, val_loss, val_loss_min):
    if val_loss <= val_loss_min:
        print("Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(val_loss_min, val_loss))
        torch.save(model, model_path)

    return min(val_loss, val_loss_min)


def log_metrics(epoch, train_loss, train_metrics, val_loss, val_metrics):
    wandb.log({'train_loss': train_loss, 'val_loss': val_loss}, step=epoch)
    wandb.log(train_metrics, step=epoch)
    wandb.log(val_metrics, step=epoch)
    print("Epoch: {}\n\t"
          "Training loss: {}\t, Training metrics: {}\n\t"
          "Validation loss: {}\t, Validation metrics: {}\n\t"
          .format(epoch, train_loss, train_metrics, val_loss, val_metrics))
