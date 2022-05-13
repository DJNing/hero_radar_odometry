import os
import argparse
import json
import torch
import numpy as np
import sys
from datasets.oxford import get_dataloaders, get_medical_loader
from datasets.boreas import get_dataloaders_boreas
from networks.under_the_radar import UnderTheRadar
# from networks.hero import HERO
from utils.utils import get_lr
from utils.losses import supervised_loss, unsupervised_loss
from utils.monitor import SVDMonitor, SteamMonitor
from datasets.transforms import augmentBatch, augmentBatch2, augmentBatch3, format_medical
import ipdb
from pathlib import Path
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(8)
torch.multiprocessing.set_sharing_strategy('file_system')
print(torch.__version__)
print(torch.version.cuda)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/steam.json', type=str, help='config file path')
    parser.add_argument('--pretrain', default=None, type=str, help='pretrain checkpoint path')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    is_medical = False
    if config['dataset'] == 'oxford':
        train_loader, valid_loader, _ = get_dataloaders(config)
    elif config['dataset'] == 'nuScenes':
        train_loader, valid_loader, _ = get_dataloaders(config)
    elif config['dataset'] == 'boreas':
        train_loader, valid_loader, _ = get_dataloaders_boreas(config)
    elif config['dataset'] == 'medical':
        is_medical = True
        config['window_size'] = 10
        train_loader, valid_loader, _ = get_medical_loader(config)

    
    if config['model'] == 'UnderTheRadar':
        model = UnderTheRadar(config).to(config['gpuid'])

    ckpt_path = None
    save_path = Path(config['log_dir'])
    save_path.mkdir(exist_ok=True)
    if os.path.isfile(str(save_path / 'latest.pt')):
        ckpt_path = str(save_path / 'latest.pt')
    elif args.pretrain is not None:
        ckpt_path = args.pretrain

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2.5e4 / config['val_rate'], factor=0.5)
    if config['model'] == 'UnderTheRadar':
        monitor = SVDMonitor(model, valid_loader, config)
    # elif config['model'] == 'HERO':
    #     monitor = SteamMonitor(model, valid_loader, config)
    start_epoch = 0

    if ckpt_path is not None:
        try:
            print('Loading from checkpoint: ' + ckpt_path)
            checkpoint = torch.load(ckpt_path, map_location=torch.device(config['gpuid']))
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            monitor.counter = checkpoint['counter']
            print('success')
        except Exception as e:
            print(e)
            print('Defaulting to legacy checkpoint style')
            model.load_state_dict(checkpoint, strict=False)
            print('success')
    if not os.path.isfile(config['log_dir'] + args.config):
        os.system('cp ' + args.config + ' ' + config['log_dir'])

    model.eval()
    
