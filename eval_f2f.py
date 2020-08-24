import signal
import argparse
import json
import os
from os import makedirs, remove
from os.path import exists, join
import time

# Dataset
from datasets.custom_sampler import *
from datasets.kitti import *
from utils.config import *
from torch.utils.data import DataLoader
import torch.nn.functional as F

# network
from networks.f2f_pose_model import F2FPoseModel

# steam
import cpp_wrappers.cpp_steam.build.steampy_f2f as steampy_f2f

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

def ransac_svd(points1, points2):
    reflect = torch.eye(3, device=points1.device)
    reflect[2, 2] = -1
    T_21 = torch.eye(4, device=points1.device)
    pick_size = 5
    total_points = points1.shape[0]
    best_inlier_size = 0

    for i in range(100):
        # randomly select query
        query = np.random.randint(0, high=total_points, size=pick_size)

        # centered
        mean1 = points1[query, :].mean(dim=0, keepdim=True)
        mean2 = points2[query, :].mean(dim=0, keepdim=True)
        diff1 = points1[query, :] - mean1
        diff2 = points2[query, :] - mean2

        # svd
        H = torch.matmul(diff1.T, diff2)
        u, s, v = torch.svd(H)
        r = torch.matmul(v, u.T)
        r_det = torch.det(r)
        if r_det < 0:
            v = torch.matmul(v, reflect)
            r = torch.matmul(v, u.T)
        R_21 = r
        t_21 = torch.matmul(-r, mean1.T) + mean2.T

        points2tran = points2@R_21 - (R_21.T@t_21).T
        error = points1 - points2tran
        error = torch.sum(error ** 2, 1)
        inliers = torch.where(error < 0.09)
        if inliers[0].size(0) > best_inlier_size:
            best_inlier_size = inliers[0].size(0)
            best_inlier_ids = inliers[0]

        if best_inlier_size > 0.8*total_points:
            break

    # svd with inliers
    query = best_inlier_ids
    mean1 = points1[query, :].mean(dim=0, keepdim=True)
    mean2 = points2[query, :].mean(dim=0, keepdim=True)
    diff1 = points1[query, :] - mean1
    diff2 = points2[query, :] - mean2

    # svd
    H = torch.matmul(diff1.T, diff2)
    u, s, v = torch.svd(H)
    r = torch.matmul(v, u.T)
    r_det = torch.det(r)
    if r_det < 0:
        v = torch.matmul(v, reflect)
        r = torch.matmul(v, u.T)
    R_21 = r
    t_21 = torch.matmul(-r, mean1.T) + mean2.T

    T_21[:3, :3] = R_21
    T_21[:3, 3:] = t_21

    return T_21, query

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='results/ir_super2_w6_mah4_p8_gtsp_sobi6/config.json', type=str,
                      help='config file path (default: config/steam_f2f.json)')

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    config['dataset']['data_dir'] = '/home/david/Data/kitti'

    # Initialize datasets
    train_dataset = KittiDataset(config, set='test')
    train_sampler = WindowBatchSampler(batch_size=1,
                                       window_size=2,
                                       seq_len=train_dataset.seq_len,
                                       drop_last=True)


    # Initialize the dataloader
    training_loader = DataLoader(train_dataset,
                                 batch_sampler=train_sampler,
                                 num_workers=0,
                                 pin_memory=True)

    # gpu
    device = torch.device("cuda:0")

    # load checkpoint
    previous_training_path = config['previous_session']
    chosen_chkp = 'chkp.tar'
    chosen_chkp = os.path.join('results', previous_training_path, chosen_chkp)
    checkpoint = torch.load(chosen_chkp)

    # network
    net = F2FPoseModel(config,
                         config['test_loader']['window_size'],
                         config['test_loader']['batch_size'])
    net.to(device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    T_21_ransac = np.zeros((train_dataset.seq_len[0] - 1, 4, 4))
    T_21_steam = np.zeros((train_dataset.seq_len[0] - 1, 4, 4))
    T_21_gt = np.zeros((train_dataset.seq_len[0] - 1, 4, 4))

    # load random pair
    for i_batch, sample_batch in enumerate(training_loader):
        fid = int(sample_batch['f_ind'][0])

        src_coords, tgt_coords, weights, patch_mask = net.forward_keypoints(sample_batch)

        # mask
        nr_ids1 = torch.nonzero(patch_mask[0, :, :].squeeze(), as_tuple=False).squeeze()
        nr_ids2 = torch.nonzero(patch_mask[1, :, :].squeeze(), as_tuple=False).squeeze()

        # get src points
        points1 = src_coords[0, :, nr_ids1].transpose(0, 1)

        # get weights
        w = weights[0, :, nr_ids1].transpose(0, 1)
        Wmat, d = net.convertWeightMat(w)

        # match consistency
        w12 = net.softmax_matcher_block.match_vals[0, nr_ids1, :] # N x M
        w12 = w12[:, nr_ids2]
        match_score, ind2to1 = torch.max(w12, dim=1)  # N
        _, ind1to2 = torch.max(w12, dim=0)  # M
        mask = torch.eq(ind1to2[ind2to1], torch.arange(ind2to1.__len__(), device=ind1to2.device))

        # thresh = 0.95
        # mask = mask*(match_score >= 0.95)

        mask_ind = torch.nonzero(mask, as_tuple=False).squeeze()
        points1 = points1[mask_ind, :]
        Wmat = Wmat[mask_ind, :, :]

        # get tgt points
        # pseudo
        # points2 = F.softmax(w12*config['networks']['pseudo_temp'], dim=1)@tgt_coords[0, :, nr_ids2].transpose(0, 1)

        # actual
        points2 = tgt_coords[0, :, nr_ids2].transpose(0, 1)
        points2 = points2[ind2to1, :]

        points2 = points2[mask_ind, :]

        # ransac
        # T_21_r, inliers = ransac_svd(points1, points2)
        # print('inliers', inliers.size())

        points2 = points2.detach().cpu().numpy()
        points1 = points1.detach().cpu().numpy()
        D = Wmat.detach().cpu().numpy()
        # inliers = inliers.detach().cpu().numpy()

        # steam
        T_21_temp = np.zeros((13, 4, 4), dtype=np.float32)
        steampy_f2f.run_steam_best_match(points1, points2, D, T_21_temp)
        T_21_steam[fid, :, :] = T_21_temp[0, :, :]

        # T_21_temp = np.zeros((13, 4, 4), dtype=np.float32)
        # steampy_f2f.run_steam_best_match(points1[inliers, :], points2[inliers, :], D[inliers, :, :], T_21_temp)
        # T_21_ransac[fid, :, :] = T_21_temp[0, :, :]

        # gt
        T_iv = sample_batch['T_iv']
        T_21 = net.se3_inv(T_iv[1, :, :])@T_iv[0, :, :]
        T_21_gt[fid, :, :] = T_21.numpy()
        print(fid)

        # save periodically
        if np.mod(fid, 200) == 0:
            np.savetxt('traj/T_21_steam.txt', np.reshape(T_21_steam[:fid, :, :], (-1, 4)), delimiter=',')
            np.savetxt('traj/T_21_gt.txt', np.reshape(T_21_gt[:fid, :, :], (-1, 4)), delimiter=',')
            np.savetxt('traj/T_21_ransac.txt', np.reshape(T_21_ransac[:fid, :, :], (-1, 4)), delimiter=',')

    np.savetxt('traj/T_21_steam.txt', np.reshape(T_21_steam, (-1, 4)), delimiter=',')
    np.savetxt('traj/T_21_gt.txt', np.reshape(T_21_gt, (-1, 4)), delimiter=',')
    np.savetxt('traj/T_21_ransac.txt', np.reshape(T_21_ransac, (-1, 4)), delimiter=',')

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)