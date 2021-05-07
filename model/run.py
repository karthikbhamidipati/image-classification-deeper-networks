import logging
from os import makedirs
from os.path import join

import torch
import wandb
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import random_split, DataLoader

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
    training = action == 'train'
    run_name = "_".join((model_key, data_key))
    config = init_wandb_session(run_name, action)

    dataset = get_dataset(data_key, root_dir, training)
    criterion = CrossEntropyLoss()
    dataset_len = len(dataset)
    model_path = join(save_path, run_name + '.pt')

    if training:
        logging.info("Training the model: {} with dataset: {}".format(model_key, data_key))
        model = NETWORKS[model_key](input_filters=dataset[0][0].shape[0], num_classes=len(dataset.classes))
        model.to(run_device)
        train_set, val_set = random_split(dataset, (int(0.8 * dataset_len), int(0.2 * dataset_len)))
        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
        optimizer = Adam(model.parameters(), lr=config.learning_rate)
        makedirs(save_path, exist_ok=True)
        train(model, train_loader, val_loader, criterion, optimizer, config.num_epochs, model_path)
    else:
        logging.info("Testing the model: {} with dataset: {}".format(model_key, data_key))
        model = torch.load(model_path)
        test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
        metrics = predict(model, test_loader, criterion)
        log_pred_metrics(metrics)
