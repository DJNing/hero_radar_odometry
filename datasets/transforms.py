import numpy as np
import cv2
import torch
from utils.utils import get_transform, get_inverse_tf

# apply a random transformation to every other frame and adjust the ground truth transform accordingly
def augmentBatch(batch, config):
    rot_max = config['augmentation']['rot_max']
    data = batch['data'].numpy()    # this seems to return a reference, not a copy
    # mask = batch['mask'].numpy()
    B, C, H, W = data.shape
    T_aug = []
    for i in range(B):
        if np.mod(i, config['window_size']) == 0:
            continue
        img = data[i].squeeze()
        # mmg = mask[i].squeeze()
        rot = np.random.uniform(-rot_max, rot_max)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rot * 180 / np.pi, 1.0)
        data[i] = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC).reshape(C, H, W)
        #mask[i] = cv2.warpAffine(mmg, M, (W, H), flags=cv2.INTER_CUBIC).reshape(C, H, W)
        T_aug += [torch.from_numpy(get_transform(0, 0, -rot))]
    batch['data'] = torch.from_numpy(data)
    # batch['mask'] = torch.from_numpy(mask > 0.5).type(batch['data'].dtype)    # make into a binary mask
    batch['T_aug'] = T_aug
    return batch
