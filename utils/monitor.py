import torch
import numpy as np
from tensorboardX import SummaryWriter
from time import time

from datasets.sequential_sampler import *

class Monitor(object):
    def __init__(self, model, valid_loader, config):
        self.model = model
        self.log_dir = config['log_dir']
        self.valid_loader = valid_loader
        self.config = config
        self.gpuid = config['gpuid']
        self.counter = 0
        self.dt = 0
        self.current_time = 0
        self.vis_batches = self.get_vis_batches()
        self.writer = SummaryWriter(self.log_dir)
        print('monitor running and saving to {}'.format(self.log_dir))

    def get_vis_batches(self):
        return np.linspace(0, len(self.valid_loader.dataset) - 1, self.config['vis_num']).astype(np.int32)

    def step(self, batchi, loss, R_loss, t_loss, batch, R_tgt_src_pred, t_tgt_src_pred):
        self.counter += 1
        self.dt = time() - self.current_time
        self.current_time = time()

        if self.counter % self.config['print_rate'] == 0:
            print('Batch: {} | Loss: {} | Step time: {}'.format(batchi, loss.detach().cpu().item(), self.dt))

        if self.counter % self.config['log_rate'] == 0:
            self.writer.add_scalar('train/loss', loss.detach().cpu().item(), self.counter)
            self.writer.add_scalar('train/R_loss', R_loss.detach().cpu().item(), self.counter)
            self.writer.add_scalar('train/t_loss', t_loss.detach().cpu().item(), self.counter)
            self.writer.add_scalar('train/step_time', self.dt, self.counter)

        if self.counter % self.config['val_rate'] == 0:
            with torch.no_grad():
                self.model.eval()
                self.validation()
                self.model.train()

        if self.counter % self.config['save_rate'] == 0:
            with torch.no_grad():
                self.model.eval()
                mname = os.path.join(self.log_dir, "{}.pt".format(self.counter))
                print('saving model', mname)
                torch.save(self.model.state_dict(), mname)
                self.model.train()

    def vis(self, batch):
        pass
        # TODO

    def validation(self):
        time_used = []
        valid_loss = 0
        valid_R_loss = 0
        valid_t_loss = 0
        for batchi, batch in enumerate(self.valid_loader):
            ts = time()
            R_tgt_src_pred, t_tgt_src_pred = model(batch)
            if (batchi + 1) % self.config['print_rate'] == 0:
                print("Eval Batch {}: {:.2}s".format(batchi, np.mean(time_used[-self.config['print_rate']:])))

            if batchi in self.vis_batches:
                self.vis(batch)

            R_tgt_src_pred, t_tgt_src_pred = model(batch)
            loss, R_loss, t_loss = supervised_loss(R_tgt_src_pred, t_tgt_src_pred, batch)
            valid_loss += loss.detach().cpu().item()
            valid_R_loss += R_loss.detach().cpu().item()
            valid_t_loss += t_loss.detach().cpu().item()
            time_used.append(time() - ts)

        self.writer.add_scalar('val/loss', valid_loss, self.counter)
        self.writer.add_scalar('val/R_loss', valid_R_loss, self.counter)
        self.writer.add_scalar('val/t_loss', valid_t_loss, self.counter)
        self.writer.add_scalar('val/avg_time_per_batch', sum(time_used)/len(time_used), self.counter)
