import logging
from os import makedirs
from os.path import join

import torch
import wandb
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model import run_device
from model.config import DATA_SOURCES, NETWORKS, PROJECT_NAME, HYPER_PARAMETERS
from model.predict import predict, log_pred_metrics
from model.train import train


def init_wandb_session(run_name, action):
    wand_run_session = wandb.init(project=PROJECT_NAME, config=HYPER_PARAMETERS)
    wand_run_session.config.update({"action": action})
    wand_run_session.name = run_name
    return wand_run_session.config


def get_dataset(data_key, root_dir, training):
    return DATA_SOURCES[data_key](root_dir, training)


def run(action, root_dir, data_key, model_key, save_path):
    run_name = "_".join((model_key, data_key))
    config = init_wandb_session(run_name, action)

    criterion = CrossEntropyLoss()
    model_path = join(save_path, run_name + '.pt')

    if action == 'train':
        logging.info("Training the model: {} with dataset: {}".format(model_key, data_key))
        train_set, val_set = get_dataset(data_key, root_dir, True)
        model = NETWORKS[model_key](input_filters=train_set[0][0].shape[0], num_classes=len(train_set.classes))
        model.to(run_device)
        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
        optimizer = SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
        scheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=config.min_learning_rate, patience=10)
        makedirs(save_path, exist_ok=True)
        train(model, train_loader, val_loader, criterion, optimizer, scheduler, config.num_epochs, model_path)
    else:
        logging.info("Testing the model: {} with dataset: {}".format(model_key, data_key))
        model = torch.load(model_path)
        test_set = get_dataset(data_key, root_dir, False)
        test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
        metrics = predict(model, test_loader, criterion)
        log_pred_metrics(metrics)
