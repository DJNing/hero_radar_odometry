# Author: Keenan Burnett
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets.custom_sampler import *
from datasets.sequential_sampler import *

def get_sequences(path, prefix='2019'):
    sequences = [f for f in os.listdir(path) if prefix in f].sort()
    return sequences

def get_frames(path, extension='.png'):
    frames = [f for f in os.listdir(path) if extension in f].sort()
    return frames

def get_frames_with_gt(frames, gt_path):
    # For the Oxford Dataset we do a search from the end backwards because some
    # of the sequences don't have GT as the end, but they all have GT at the beginning.
    frames_out = frames
    with open(gt_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(frames) - 1, -1, -1):
            frame = frames[i].split('.')[0]
            if frame not in lines:
                frames_out.pop()
            else:
                break
    return frames_out

def get_inverse_tf(T):
    T2 = np.identity(4)
    R = T[0:3, 0:3]
    t = T[0:3, 3].reshape(3, 1)
    T2[0:3, 0:3] = R.transpose()
    t = np.matmul(-1 * R.transpose(), t)
    T2[0:3, 3] = np.squeeze(t)
    return T2

def get_transform(x, y, theta):
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    T = np.identity(4)
    T[0:2, 0:2] = R
    T[0, 3] = x
    T[1, 3] = y
    return T

def get_groundtruth_odometry(radar_time, gt_path):
    with open(gt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            if int(line[9]) == radar_time:
                T = get_transform(float(line[2]), float(line[3]), float(line[7])) # T_1_2 (from next time to current)
                return get_inverse_tf(T) # T_2_1 (from current time step to the next time step)
    assert(0), "ground truth transform not found"

class OxfordDataset(Dataset):
    """Oxford Radar Robotcar Dataset """

    def __init__(self, config, split='train'):
        self.config = config
        self.data_dir = config['data_dir']
        sequences = get_sequences(self.data_dir)
        self.sequences = get_sequences_split(sequences, split)
        self.seq_idx_range = []
        self.frames = []
        self.seq_len = []
        for seq in self.sequences:
            seq_frames = get_frames(data_dir + seq + "/radar/")
            seq_frames = get_frames_with_gt(seq_frames, data_dir + seq + "/gt/radar_odometry.csv")
            self.seq_idx_range[seq] = [len(frames), len(frames) + len(seq_frames)]
            self.seq_len.append(len(seq_frames))
            self.frames.extend(seq_frames)

    def get_sequences_split(sequences, split):
        self.split = config['train_split']
        if split == 'validation':
            self.split = config['validation_split']
        elif split == 'test':
            self.split = config['test_split']
        return [i for i in range(0, len(sequences)) if self.split[0] <= i and i < self.split[1]]

    def __len__(self):
        return len(self.frames)

    def get_seq_from_idx(idx):
        for seq in self.sequences:
            if seq_idx_range[seq][0] <= idx and idx < seq_idx_range[seq][1]:
                return seq
        assert(0), "sequence for this idx not found"

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        seq = get_seq_from_idx(idx)
        frame = self.data_dir + seq + "/radar/" + self.frames[idx]
        timestamps, azimuths, _, fft_data, _ = load_radar(frame)
        cart = radar_polar_to_cartesian(azimuths, fft_data, config['radar_resolution'], config['cart_resolution'],
            config['cart_pixel_width'])
        time = int(self.frames[idx].split('.')[0])
        # Get ground truth transform between this frame and the next
        transform = get_groundtruth_odometry(time, self.data_dir + seq + "/gt/radar_odometry.csv")
        return {'input': cart, 'T_21': transform}

def get_dataloader(config):
    vconfig = config
    vconfig['batch_size'] = 1
    train_data = OxfordDataset(config, 'train')
    valid_data = OxfordDataset(vconfig, 'validation')
    train_sampler = RandomWindowBatchSampler(config['batch_size'], config['window_size'], train_dataset.seq_len)
    valid_sampler = SequentialWindowBatchSampler(batch_size = 1, config['window_size'], valid_dataset.seq_len)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=config['num_workers'])
    valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=config['num_workers'])
    return train_loader, valid_loader
