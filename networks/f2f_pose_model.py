import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import zn_desc, T_inv
from networks.unet_block import UNetBlock
from networks.softmax_matcher_block import SoftmaxMatcherBlock
from networks.svd_weight_block import SVDWeightBlock
from networks.svd_block import SVDBlock
from networks.keypoint_block import KeypointBlock

class F2FPoseModel(nn.Module):
    def __init__(self, config, window_size, batch_size):
        super(F2FPoseModel, self).__init__()

        # load configs
        self.config = config
        self.window_size = window_size # TODO this is hard-fixed at the moment
        self.match_type = config["networks"]["match_type"] # zncc, l2, dp

        # network arch
        self.unet_block = UNetBlock(self.config)

        self.keypoint_block = KeypointBlock(self.config, window_size, batch_size)

        self.softmax_matcher_block = SoftmaxMatcherBlock(self.config)

        self.svd_weight_block = SVDWeightBlock(self.config)

        self.svd_block = SVDBlock(self.config)

    def forward(self, data):
        '''
        Estimate transform between two frames
        :param data: dictionary containing outputs of getitem function
        :return:
        '''

        # parse data
        geometry_img, images, T_iv = data['geometry'], data['input'], data['T_iv']

        # move to GPU
        geometry_img = geometry_img.cuda()
        images = images.cuda()
        T_iv = T_iv.cuda()

        # divide range by 100
        images[1, :, :] = images[1, :, :]/100.0

        # Extract features, detector scores and weight scores
        detector_scores, weight_scores, descs = self.unet_block(images)

        # Use detector scores to compute keypoint locations in 3D along with their weight scores and descs
        keypoint_coords, keypoint_descs, keypoint_weights = self.keypoint_block(geometry_img, descs, detector_scores, weight_scores)

        # Match the points in src frame to points in target frame to generate pseudo points
        # first input is src. Computes pseudo with target
        pseudo_coords, pseudo_weights, pseudo_descs = self.softmax_matcher_block(keypoint_coords[::self.window_size],
                                                                                 keypoint_coords[1::self.window_size],
                                                                                 keypoint_weights[1::self.window_size],
                                                                                 keypoint_descs[::self.window_size],
                                                                                 keypoint_descs[1::self.window_size])

        # compute loss
        loss = self.loss(keypoint_coords[::self.window_size], pseudo_coords,
                         keypoint_weights[::self.window_size], T_iv)


        return loss

    def loss(self, src_coords, tgt_coords, weights, T_iv):
        '''
        Compute loss
        :param src_coords: src keypoint coordinates
        :param tgt_coords: tgt keypoint coordinates
        :param weights: weights for match pair
        :param T_iv: groundtruth transform
        :return:
        '''
        loss = 0

        # loop over each batch
        for batch_i in range(src_coords.size(0)):
            b = 1
            # get src points
            points1 = src_coords[batch_i, :, :].transpose(0, 1)

            # check for no returns in src keypoints
            nr_ids = torch.nonzero(torch.sum(points1, dim=1), as_tuple=False).squeeze()
            points1 = points1[nr_ids, :]

            # get tgt points
            points2 = tgt_coords[batch_i, :, nr_ids].transpose(0, 1)

            # get weights
            w = weights[batch_i, :, nr_ids].transpose(0, 1)

            # get gt poses
            src_id = 2*batch_i
            tgt_id = 2*batch_i + 1
            T_21 = self.se3_inv(T_iv[tgt_id, :, :])@T_iv[src_id, :, :]

            # error rejection
            points1_in_2 = points1@T_21[:3, :3].T + T_21[:3, 3].unsqueeze(0)
            error = torch.sum((points1_in_2 - points2) ** 2, dim=1)
            ids = torch.nonzero(error < self.config["networks"]["keypoint_loss"]["error_thresh"] ** 2,
                                as_tuple=False).squeeze()

            loss += self.weighted_mse_loss(points1_in_2[ids, :],
                                           points2[ids, :],
                                           w[ids, :])

            loss -= torch.mean(3*w[ids, :])

        return loss

    def weighted_mse_loss(self, data, target, weight):
        return 3.0*torch.mean(torch.exp(weight) * (data - target) ** 2)

    def print_loss(self, epoch, iter, loss):
        message = '{:d},{:d},{:.6f}'.format(epoch, iter, loss)
        print(message)

    def se3_inv(self, Tf):
        Tinv = torch.zeros_like(Tf)
        Tinv[:3, :3] = Tf[:3, :3].T
        Tinv[:3, 3:] = -Tf[:3, :3].T@Tf[:3, 3:]
        Tinv[3, 3] = 1
        return Tinv
