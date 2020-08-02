import os
import signal
import argparse
import json

import torch.nn as nn
from torch.utils.data import DataLoader

from utils.trainer import Trainer
from datasets.custom_sampler import *
from datasets.kitti import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/sample.json', type=str,
                      help='config file path (default: config/sample.json)')

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # dataloader setup
    train_dataset = KittiDataset(config, set='training')
    train_sampler = RandomWindowBatchSampler(batch_size=config["data_loader"]["batch_size"],
                                             window_size=config["data_loader"]["window_size"],
                                             seq_len=train_dataset.seq_len,
                                             drop_last=True)
    train_loader = DataLoader(train_dataset,
                              batch_sampler=train_sampler,
                              num_workers=config["data_loader"]["num_workers"],
                              pin_memory=True)

    valid_dataset = KittiDataset(config, set='validation')
    valid_sampler = RandomWindowBatchSampler(batch_size=config["data_loader"]["batch_size"],
                                             window_size=config["data_loader"]["window_size"],
                                             seq_len=valid_dataset.seq_len,
                                             drop_last=True)
    valid_loader = DataLoader(train_dataset,
                              batch_sampler=valid_sampler,
                              num_workers=config["data_loader"]["num_workers"],
                              pin_memory=True)

    # network setup (dummy network for now)
    model = nn.Sequential(
            nn.Conv2d(1,20,5),
            nn.ReLU(),
            nn.Conv2d(20,64,5),
            nn.ReLU())

    # trainer
    trainer = Trainer(model, train_loader, valid_loader, config)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)

