"""
    PyTorch dataset class for the Oxford Radar Robotcar Dataset.
    Authors: Keenan Burnett
"""
from glob import glob
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets.custom_sampler import RandomWindowBatchSampler, SequentialWindowBatchSampler
from datasets.radar import load_radar, radar_polar_to_cartesian
from datasets.interpolate_poses import interpolate_ins_poses
from utils.utils import get_inverse_tf, get_transform
import cv2
from pathlib import Path

import ipdb

def get_transform_oxford(x, y, theta):
    """Returns a 4x4 homogeneous 3D transform for given 2D parameters (x, y, theta).
    Note: (x,y) are position of frame 2 wrt frame 1 as measured in frame 1.
    Args:
        x (float): x translation
        x (float): y translation
        theta (float): rotation
    Returns:
        np.ndarray: 4x4 transformation matrix from next time to current (T_1_2)
    """
    T = np.identity(4, dtype=np.float32)
    T[0:2, 0:2] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    T[0, 3] = x
    T[1, 3] = y
    return T

def get_sequences(path, prefix='2019'):
    """Retrieves a list of all the sequences in the dataset with the given prefix.
        Sequences are subfolders underneath 'path'
    Args:
        path (AnyStr): path to the root data folder
        prefix (AnyStr): each sequence / subfolder must begin with this common string.
    Returns:
        List[AnyStr]: List of sequences / subfolder names.
    """
    sequences = [f for f in os.listdir(path) if prefix in f]
    sequences.sort()
    return sequences

def get_frames(path, extension='.png'):
    """Retrieves all the file names within a path that match the given extension.
    Args:
        path (AnyStr): path to the root/sequence/sensor/ folder
        extension (AnyStr): each data frame must end with this common string.
    Returns:
        List[AnyStr]: List of frames / file names.
    """
    frames = [f for f in os.listdir(path) if extension in f]
    frames.sort()
    return frames

def mean_intensity_mask(polar_data, multiplier=3.0):
    """Thresholds on multiplier*np.mean(azimuth_data) to create a polar mask of likely target points.
    Args:
        polar_data (np.ndarray): num_azimuths x num_range_bins polar data
        multiplier (float): multiple of mean that we treshold on
    Returns:
        np.ndarray: binary polar mask corresponding to likely target points
    """
    num_azimuths, range_bins = polar_data.shape
    mask = np.zeros((num_azimuths, range_bins))
    for i in range(num_azimuths):
        m = np.mean(polar_data[i, :])
        mask[i, :] = polar_data[i, :] > multiplier * m
    return mask

class OxfordDataset(Dataset):
    """Oxford Radar Robotcar Dataset."""
    def __init__(self, config, split='train'):
        self.config = config
        self.data_dir = config['data_dir']
        dataset_prefix = ''
        if config['dataset'] == 'oxford':
            dataset_prefix = '2019'
        elif config['dataset'] == 'boreas':
            dataset_prefix = 'boreas'
        elif config['dataset'] == 'nuScenes':
            dataset_prefix = 'nuScenes'
        self.T_radar_imu = np.eye(4, dtype=np.float32)
        for i in range(3):
            self.T_radar_imu[i, 3] = config['steam']['ex_translation_vs_in_s'][i]
        sequences = get_sequences(self.data_dir, dataset_prefix)
        self.sequences = self.get_sequences_split(sequences, split)
        self.seq_idx_range = {}
        self.frames = []
        self.seq_lens = [] # list of int
        for seq in self.sequences:
            seq_path = os.path.join(self.data_dir, seq)
            seq_frames = get_frames(seq_path + '/radar/')
            seq_frames = self.get_frames_with_gt(seq_frames, seq_path + '/gt/radar_odometry.csv')
            self.seq_idx_range[seq] = [len(self.frames), len(self.frames) + len(seq_frames)]
            self.seq_lens.append(len(seq_frames))
            self.frames.extend(seq_frames)

    def get_sequences_split(self, sequences, split):
        """Retrieves a list of sequence names depending on train/validation/test split.
        Args:
            sequences (List[AnyStr]): list of all the sequences, sorted lexicographically
            split (List[int]): indices of a specific split (train or val or test) aftering sorting sequences
        Returns:
            List[AnyStr]: list of sequences that belong to the specified split
        """
        if self.config['dataset'] == 'nuScenes':

            train_ratio, val_ratio, test_ratio = self.config['train_val_test_ratio']

            total_len = len(sequences)

            val_start = int(train_ratio * total_len)
            test_start = int((train_ratio+val_ratio) * total_len)

            if split == 'train':
                self.split = list(range(val_start))
                return sequences[:val_start]
            elif split == 'validation':
                self.split = list(range(val_start,test_start))
                return sequences[val_start:test_start]
            elif split == 'test':
                self.split = self.config['test_split']
                return [seq for i, seq in enumerate(sequences) if i in self.split]
        else:
            self.split = self.config['train_split']
            if split == 'validation':
                self.split = self.config['validation_split']
            elif split == 'test':
                self.split = self.config['test_split']
            return [seq for i, seq in enumerate(sequences) if i in self.split]

    def get_frames_with_gt(self, frames, gt_path, backward=True):
        """Retrieves the subset of frames that have groundtruth
        Note: For the Oxford Dataset we do a search from the end backwards because some
            of the sequences don't have GT as the end, but they all have GT at the beginning.
        Args:
            frames (List[AnyStr]): List of file names
            gt_path (AnyStr): path to the ground truth csv file
        Returns:
            List[AnyStr]: List of file names with ground truth
        """
        def check_if_frame_has_gt(frame, gt_lines):
            for i in range(len(gt_lines) - 1, -1, -1):
                line = gt_lines[i].split(',')
                if frame == int(line[9]):
                    return True
            return False
        frames_out = frames
        with open(gt_path, 'r') as f:
            f.readline()
            lines = f.readlines()
            for i in range(len(frames) - 1, -1, -1):
                frame = int(frames[i].split('.')[0])
                if check_if_frame_has_gt(frame, lines):
                    if backward:
                        break
                    # else:
                    #     continue
                frames_out.pop()
                # else:
                #     frames_out.remove(frames[i])
        return frames_out

    def get_groundtruth_odometry(self, radar_time, gt_path):
        """Retrieves the groundtruth 4x4 transform from current time to next
        Args:
            radar_time (int): UNIX INT64 timestamp that we want groundtruth for (also the filename for radar)
            gt_path (AnyStr): path to the ground truth csv file
        Returns:
            T_2_1 (np.ndarray): 4x4 transformation matrix from current time to next
            time1 (int): UNIX INT64 timestamp of the current frame
            time2 (int): UNIX INT64 timestamp of the next frame
        """
        with open(gt_path, 'r') as f:
            f.readline()
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.split(',')
                if int(line[9]) == radar_time:
                    T = get_transform_oxford(float(line[2]), float(line[3]), float(line[7]))  # from next time to current
                    return T, int(line[1]), int(line[0]) # T_2_1 from current time step to the next
        assert(0), 'ground truth transform for {} not found in {}'.format(radar_time, gt_path)

    def get_groundruth_ins(self, time1, time2, gt_path):
        """Extracts ground truth transform T_2_1 from INS data, from current time1 to time2
        Args:
            time1 (int): UNIX INT64 timestamp of the current frame
            time2 (int): UNIX INT64 timestamp of the next frame
            gt_path (AnyStr): path to the ground truth csv file
        Returns:
            T_2_1 (np.ndarray): 4x4 transformation matrix from current time to next
        """
        T = np.array(interpolate_ins_poses(gt_path, [time1], time2)[0])
        return self.T_radar_imu @ T @ get_inverse_tf(self.T_radar_imu)

    def __len__(self):
        return len(self.frames)

    def get_seq_from_idx(self, idx):
        """Returns the name of the sequence that this idx belongs to.
        Args:
            idx (int): frame index in dataset
        Returns:
            AnyStr: name of the sequence that this idx belongs to
        """
        for seq in self.sequences:
            if self.seq_idx_range[seq][0] <= idx and idx < self.seq_idx_range[seq][1]:
                return seq
        assert(0), 'sequence for idx {} not found in {}'.format(idx, self.seq_idx_range)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        seq = self.get_seq_from_idx(idx)
        frame = self.data_dir +  seq + '/radar/' + self.frames[idx]
        timestamps, azimuths, _, polar = load_radar(frame)
        data = radar_polar_to_cartesian(azimuths, polar, self.config['radar_resolution'],
                                        self.config['cart_resolution'], self.config['cart_pixel_width'])  # 1 x H x W
        polar_mask = mean_intensity_mask(polar)
        mask = radar_polar_to_cartesian(azimuths, polar_mask, self.config['radar_resolution'],
                                        self.config['cart_resolution'],
                                        self.config['cart_pixel_width']).astype(np.float32)
        # Get ground truth transform between this frame and the next
        radar_time = int(self.frames[idx].split('.')[0])

        T_21, time1, time2 = self.get_groundtruth_odometry(radar_time, self.data_dir + seq + '/gt/radar_odometry.csv')
        if self.config['use_ins']:
            #T_21 = np.array(self.get_groundruth_ins(time1, time2, self.data_dir + '/' + seq + '/gps/ins.csv'))
            T_21, time1, time2 = self.get_groundtruth_odometry(radar_time, self.data_dir + seq + '/gt/radar_odometry_ins.csv')
        t_ref = np.array([time1, time2]).reshape(1, 2)
        polar = np.expand_dims(polar, axis=0)
        azimuths = np.expand_dims(azimuths, axis=0)
        timestamps = np.expand_dims(timestamps, axis=0)
        return {'data': data, 'T_21': T_21, 't_ref': t_ref, 'mask': mask, 'polar': polar, 'azimuths': azimuths,
                'timestamps': timestamps}

class nuScenesDataset(OxfordDataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.data_dir = config['data_dir']
        dataset_prefix = ''
        if config['dataset'] == 'oxford':
            dataset_prefix = '2019'
        elif config['dataset'] == 'boreas':
            dataset_prefix = 'boreas'
        elif config['dataset'] == 'nuScenes':
            dataset_prefix = 'nuScenes'
        # self.T_radar_imu = np.eye(4, dtype=np.float32)
        # for i in range(3):
        #     self.T_radar_imu[i, 3] = config['steam']['ex_translation_vs_in_s'][i]
        sequences = get_sequences(self.data_dir, dataset_prefix)
        # ipdb.set_trace()
        self.sequences = self.get_sequences_split(sequences, split)
        self.seq_idx_range = {}
        self.frames = []
        self.seq_lens = []
        for seq in self.sequences:
            seq_path = os.path.join(self.data_dir, seq)
            seq_frames = get_frames(seq_path + '/radar/')
            seq_frames = self.get_frames_with_gt(seq_frames, seq_path + '/gt/radar_odometry.csv')
            # ipdb.set_trace()
            self.seq_idx_range[seq] = [len(self.frames), len(self.frames) + len(seq_frames)]
            # ipdb.set_trace()
            self.seq_lens.append(len(seq_frames))
            self.frames.extend(seq_frames)
            # ipdb.set_trace()
            # pass
        # print("frames check")
        # print(sum(self.seq_lens) == len(self.frames))
        # print("seq_lens = ", str(sum(self.seq_lens)))
        # print("frames = ", str(len(self.frames)))


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        seq = self.get_seq_from_idx(idx)
        frame = self.data_dir + seq + '/radar/' + self.frames[idx]
        data = cv2.imread(frame)[:,:,0:1]
        data = np.transpose(data, [2,0,1])
        # data = torch.from_numpy(data)
        # timestamps, azimuths, _, polar = load_radar(frame)
        # data = radar_polar_to_cartesian(azimuths, polar, self.config['radar_resolution'],
        #                                 self.config['cart_resolution'], self.config['cart_pixel_width'])  # 1 x H x W
        # polar_mask = mean_intensity_mask(polar)
        # mask = radar_polar_to_cartesian(azimuths, polar_mask, self.config['radar_resolution'],
        #                                 self.config['cart_resolution'],
        #                                 self.config['cart_pixel_width']).astype(np.float32)
        # Get ground truth transform between this frame and the next
        radar_time = int(self.frames[idx].split('.')[0])

        T_21, time1, time2 = self.get_groundtruth_odometry(radar_time, self.data_dir + seq + '/gt/radar_odometry.csv')
        # if self.config['use_ins']:
        #     #T_21 = np.array(self.get_groundruth_ins(time1, time2, self.data_dir + '/' + seq + '/gps/ins.csv'))
        #     T_21, time1, time2 = self.get_groundtruth_odometry(radar_time, self.data_dir + '/' + seq + '/gt/radar_odometry_ins.csv')
        t_ref = np.array([time1, time2]).reshape(1, 2)
        # polar = np.expand_dims(polar, axis=0)
        # azimuths = np.expand_dims(azimuths, axis=0)
        timestamps = radar_time
        timestamps = np.expand_dims(timestamps, axis=0)
        # result = {'data': torch.from_numpy(data), 'T_21': torch.from_numpy(T_21), 't_ref': torch.from_numpy(t_ref), 'mask': torch.from_numpy(data), 'polar': None, 'azimuths': None,
        #         'timestamps': torch.from_numpy(timestamps)}
        data = torch.from_numpy(data).type(torch.FloatTensor)
        T_21 = torch.from_numpy(T_21).type(torch.FloatTensor)
        # need mask for data augmentation, since our input is binary, mask == data
        return {'data': data, 'mask': data, 'T_21': T_21, 't_ref':t_ref}

class medicalDataset(Dataset):
    def __init__(self, config, split='train') -> None:
        self.config = config
        self.data_dir = Path(config['data_dir'])
        self.frames = glob(str(self.data_dir / '*.png'))
        self.split = split
        split_ratio = [0.65, 0.15, 0.2]
        if split.lower() == 'train':
            start_idx = 0
            end_idx = int(len(self.frames) * split_ratio[0])
        elif split.lower() == 'validation':
            start_idx = int(len(self.frames) * split_ratio[0])
            end_idx = int(len(self.frames) * (split_ratio[0] + split_ratio[1]))
        elif split.lower() == 'test':
            start_idx = int(len(self.frames) * (split_ratio[0] + split_ratio[1]))
            end_idx = int(len(self.frames) * (split_ratio[1] + split_ratio[2]))
        else:
            raise RuntimeError('split %s not implemented' % split)
        self.window_size = config['window_size']
        self.frames = self.frames[start_idx:end_idx]
        self.seq_lens = [len(self.frames)]
        self.sequences = ['00']

    def __len__(self):
        return len(self.frames)

    @staticmethod
    def get_skew(x):
        return np.array([[0, -x[2], x[1]],
                        [x[2], 0, -x[0]],
                        [-x[1], x[0], 0]])

    def augment_image(self, img):
        rot_max = self.config['augmentation']['rot_max']
        tr_max = self.config['augmentation']['tr_max'] # assuming this comes with quantity as pixel
        rot = np.random.uniform(-rot_max, rot_max)
        tr_x = np.random.uniform(-tr_max, tr_max)
        tr_y = np.random.uniform(-tr_max, tr_max)
        T = get_transform(tr_x, tr_y, rot)
        # T_tr = get_transform(tr_x, tr_y, 0)
        _, H, W = img.shape
        # img_rot = 
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rot * 180 / np.pi, 1.0)
        aug_img = cv2.warpAffine(img[0,:,:], M, (W, H), flags=cv2.INTER_CUBIC).reshape(1, H, W)
        return aug_img, T

    def __getitem__(self, idx):
        base_img = cv2.imread(self.frames[idx])
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        base_img = np.expand_dims(base_img, 0) # [1, h, w]
        img_list = [base_img]
        T_list = []
        cur_img = base_img
        # generate 
        for i in range(self.window_size):
            temp_img, temp_T = self.augment_image(cur_img)
            temp_T = np.expand_dims(temp_T, 0)
            img_list += [temp_img]
            T_list += [temp_T]
            cur_img = temp_img
        # T_list += [temp_T]
        img_list = img_list[:-1]
        batch_img = np.concatenate(img_list, axis=0)
        batch_img = torch.from_numpy(batch_img).type(torch.FloatTensor)
        batch_T = np.concatenate(T_list, axis=0)
        batch_T = torch.from_numpy(batch_T).type(torch.FloatTensor)
        return {'data': batch_img, 'mask':[], 'T_21':batch_T, 't_ref':[]}
        


def get_dataloaders(config):
    """Returns the dataloaders for training models in pytorch.
    Args:
        config (json): parsed configuration file
    Returns:
        train_loader (DataLoader)
        valid_loader (DataLoader)
        test_loader (DataLoader)
    """
    vconfig = dict(config)
    vconfig['batch_size'] = 1
    if config['dataset'] == 'oxford':
        train_dataset = OxfordDataset(config, 'train')
        valid_dataset = OxfordDataset(vconfig, 'validation')
        test_dataset = OxfordDataset(vconfig, 'test')
    elif config['dataset'] == 'nuScenes':
        train_dataset = nuScenesDataset(config, 'train')
        valid_dataset = nuScenesDataset(vconfig, 'validation')
        test_dataset = nuScenesDataset(vconfig, 'test')
    train_sampler = RandomWindowBatchSampler(config['batch_size'], config['window_size'], train_dataset.seq_lens)
    valid_sampler = SequentialWindowBatchSampler(1, config['window_size'], valid_dataset.seq_lens)
    test_sampler = SequentialWindowBatchSampler(1, config['window_size'], test_dataset.seq_lens)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=config['num_workers'])    
    valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=config['num_workers'])
    # reset seq_lens after sampler
    window_size = config['window_size']
    temp_seq_lens = valid_loader.dataset.seq_lens
    new_seq_lens = [x - window_size + 1 for x in temp_seq_lens]
    valid_loader.dataset.seq_lens = new_seq_lens
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=config['num_workers'])
    temp_seq_lens = test_loader.dataset.seq_lens
    new_seq_lens = [x - window_size + 1 for x in temp_seq_lens]
    test_loader.dataset.seq_lens = new_seq_lens
    return train_loader, valid_loader, test_loader

def get_medical_loader(config):
    vconfig = dict(config)
    vconfig['batch_size'] = 1
    train_dataset = medicalDataset(config, 'train')
    valid_dataset = medicalDataset(vconfig, 'validation')
    test_dataset = medicalDataset(vconfig, 'test')

    train_loder = DataLoader(train_dataset, num_workers=config['num_workers'])
    valid_loder = DataLoader(valid_dataset, num_workers=config['num_workers'])
    test_loder = DataLoader(test_dataset, num_workers=config['num_workers'])

    return train_loder, valid_loder, test_loder