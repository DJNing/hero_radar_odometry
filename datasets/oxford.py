import os
import torch
import numpy as np
from torch.utils.data import Dataset, Dataloader

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

class OxfordDataset(Dataset):
    """Oxford Radar Robotcar Dataset """

    def __init__(self, config, split='train'):
        self.config = config
        self.data_dir = config['data_dir']
        sequences = get_sequences(data_dir)
        self.sequences = get_sequences_split(sequences, split)
        self.seq_idx_range = []
        self.frames = []
        for seq in self.sequences:
            seq_frames = get_frames(data_dir + seq + "/radar/")
            # Make sure each frame has corresponding ground truth
            seq_frames = get_frames_with_gt(seq_frames, data_dir + seq + "/gt/radar_odometry.csv")
            seq_frames = seq_frames[:-config['window_size']]
            self.seq_idx_range[seq] = [len(frames), len(frames) + len(seq_frames)]
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
                return req
        assert(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        seq = get_seq_from_idx(idx)
        frame = self.data_dir + seq + "/radar/" + self.frames[idx]
        timestamps, azimuths, _, fft_data, _ = load_radar(frame)
        cart = radar_polar_to_cartesian(azimuths, fft_data, config['radar_resolution'], config['cart_resolution'],
            config['cart_pixel_width'])
        # Get ground truth transform between this frame and the next
        sample = {'radar': cart, 'transform': transform}