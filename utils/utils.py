import pickle
import numpy as np
import torch

def supervised_loss(R_tgt_src_pred, t_tgt_src_pred, batch, config):
    T_21 = batch['T_21'].to(config['gpuid'])
    # Get ground truth transforms
    kp_inds, _ = get_indices(config['batch_size'], config['window_size'])
    T_tgt_src = T_21[kp_inds]
    R_tgt_src = T_tgt_src[:, :3, :3]
    t_tgt_src = T_tgt_src[:, :3, 3].unsqueeze(-1)
    svd_loss, R_loss, t_loss = SVD_loss(R_tgt_src, R_tgt_src_pred, t_tgt_src, t_tgt_src_pred, config['gpuid'])
    dict_loss = {'R_loss': R_loss, 't_loss': t_loss}
    return svd_loss, dict_loss

def pointmatch_loss(out, batch, config, alpha=1.0, beta=5.0, errorT=100.0):
    R_tgt_src_pred = out['R']
    t_tgt_src_pred = out['t']
    tgt = out['tgt']
    src = out['src']
    weights = out['match_weights']
    dense_weights = out['dense_weights']
    mask = batch['mask'].to(config['gpuid'])
    # tgt, src: B x N x 2
    assert(tgt.size() == src.size())
    B, N, _ = tgt.size()
    R = R_tgt_src_pred[:, :2, :2]  # B x 2 x 2
    t = t_tgt_src_pred[:, :2].expand(B, 2, N)  # B x 2 x N
    tgt_pred = (torch.bmm(R, src.transpose(2, 1)) + t).transpose(2, 1) # B x N x 2
    l1loss = torch.nn.L1Loss()
    point_loss = l1loss(tgt, tgt_pred)
    dict_loss = {'point_loss': point_loss}
    wsum = torch.sum(weights)
    if wsum == 0:
        weight_loss = 15
        print('WARNING: matching weights have gone to zero!')
    else:
        weight_loss = -1 * torch.log(wsum / (B * N))
    dict_loss['weight_loss'] = weight_loss
    #bceloss = torch.nn.BCELoss()
    #mask_loss = bceloss(dense_weights, mask)
    #dict_loss['mask_loss'] = mask_loss
    loss = point_loss + alpha * weight_loss #+ beta * mask_loss
    return loss, dict_loss

def SVD_loss(R, R_pred, t, t_pred, gpuid='cpu', alpha=10.0):
    batch_size = R.size(0)
    identity = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(gpuid)
    loss_fn = torch.nn.L1Loss()
    R_loss = alpha * loss_fn(torch.matmul(R_pred.transpose(2, 1), R), identity)
    t_loss = 1.0 * loss_fn(t_pred, t)
    svd_loss = R_loss + t_loss
    return svd_loss, R_loss, t_loss

def get_inverse_tf(T):
    """Returns the inverse of a given 4x4 homogeneous transform."""
    T2 = np.identity(4, dtype=np.float32)
    R = T[0:3, 0:3]
    t = T[0:3, 3].reshape(3, 1)
    T2[0:3, 0:3] = R.transpose()
    t = np.matmul(-1 * R.transpose(), t)
    T2[0:3, 3] = np.squeeze(t)
    return T2

def get_transform(x, y, theta):
    """Returns a 4x4 homogeneous 3D transform for a given 2D (x, y, theta)."""
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    T = np.identity(4, dtype=np.float32)
    T[0:2, 0:2] = R
    T[0, 3] = x
    T[1, 3] = y
    return T

def get_transform2(R, t):
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t.squeeze()
    return T

def enforce_orthog(T, dim=3):
    """Enforces the orthogonality of a 3x3 rotation matrix within a 4x4 homogeneous transformation matrix."""
    if dim == 2:
        if abs(np.linalg.det(T[0:2, 0:2]) - 1) < 1e-10:
            return T
        R = T[0:2, 0:2]
        epsilon = 0.001
        if abs(R[0, 0] - R[1, 1]) > epsilon or abs(R[1, 0] + R[0, 1]) > epsilon:
            print("WARNING: this is not a proper rigid transformation:", R)
            return T
        a = (R[0, 0] + R[1, 1]) / 2
        b = (-R[1, 0] + R[0, 1]) / 2
        sum = np.sqrt(a**2 + b**2)
        a /= sum
        b /= sum
        R[0, 0] = a; R[0, 1] = b
        R[1, 0] = -b; R[1, 1] = a
        T[0:2, 0:2] = R
    if dim == 3:
        if abs(np.linalg.det(T[0:3, 0:3]) - 1) < 1e-10:
            return T
        c1 = T[0:3, 1]
        c2 = T[0:3, 2]
        c1 /= np.linalg.norm(c1)
        c2 /= np.linalg.norm(c2)
        newcol0 = np.cross(c1, c2)
        newcol1 = np.cross(c2, newcol0)
        T[0:3, 0] = newcol0
        T[0:3, 1] = newcol1
        T[0:3, 2] = c2
    return T

# Use axis-angle representation to get a single number for rotation error
def rotationError(T):
    d = 0.5 * (np.trace(T[0:3, 0:3]) - 1)
    return np.arccos(max(min(d, 1.0), -1.0))

def translationError(T, dim=2):
    if dim == 2:
        return np.sqrt(T[0, 3]**2 + T[1, 3]**2)
    if dim == 3:
        return np.sqrt(T[0, 3]**2 + T[1, 3]**2 + T[2, 3]**2)

def computeMedianError(T_gt, T_pred):
    """Computes the median translation and rotation error along with their standard deviations."""
    t_error = []
    r_error = []
    for i, T in enumerate(T_gt):
        T_error = np.matmul(T, get_inverse_tf(T_pred[i]))
        t_error.append(translationError(T_error))
        r_error.append(180 * rotationError(T_error) / np.pi)
    t_error = np.array(t_error)
    r_error = np.array(r_error)
    return [np.median(t_error), np.std(t_error), np.median(r_error), np.std(r_error), np.mean(t_error), np.mean(r_error)]

def trajectoryDistances(poses):
    """Calculates path length along the trajectory."""
    dist = [0]
    for i in range(1, len(poses)):
        P1 = get_inverse_tf(poses[i - 1])
        P2 = get_inverse_tf(poses[i])
        dx = P1[0, 3] - P2[0, 3]
        dy = P1[1, 3] - P2[1, 3]
        dist.append(dist[i-1] + np.sqrt(dx**2 + dy**2))
    return dist

def lastFrameFromSegmentLength(dist, first_frame, length):
    for i in range(first_frame, len(dist)):
        if dist[i] > dist[first_frame] + length:
            return i
    return -1

def calcSequenceErrors(poses_gt, poses_pred):
    lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    err = []
    step_size = 4  # Every second
    # Pre-compute distances from ground truth as reference
    dist = trajectoryDistances(poses_gt)

    for first_frame in range(0, len(poses_gt), step_size):
        for length in lengths:
            last_frame = lastFrameFromSegmentLength(dist, first_frame, length)
            if last_frame == -1:
                continue
            # Compute rotational and translation errors
            pose_delta_gt = np.matmul(poses_gt[last_frame], get_inverse_tf(poses_gt[first_frame]))
            pose_delta_res = np.matmul(poses_pred[last_frame], get_inverse_tf(poses_pred[first_frame]))
            pose_error = np.matmul(pose_delta_gt, get_inverse_tf(pose_delta_res))
            r_err = rotationError(pose_error)
            t_err = translationError(pose_error)
            # Approx speed
            num_frames = float(last_frame - first_frame + 1)
            speed = float(length) / (0.25 * num_frames)
            err.append([first_frame, r_err/float(length), t_err/float(length), length, speed])
    return err

def getStats(err):
    t_err = 0
    r_err = 0
    for e in err:
        t_err += e[2]
        r_err += e[1]
    t_err /= float(len(err))
    r_err /= float(len(err))
    return t_err, r_err

def computeKittiMetrics(T_gt, T_pred, seq_lens):
    """
        Computes the translational (%) and rotational drift (deg/m) in the KITTI style.
        T_gt: List of 4x4 homogeneous transforms (Frame t to Frame t+1)
        T_pred: List of 4x4 homogeneous transforms (Frame t to Frame t+1)
        seq_lens: List of sequence lengths
    """
    seq_indices = []
    idx = 0
    for s in seq_lens:
        seq_indices.append(list(range(idx, idx + s - 1)))
        idx += (s - 1)
    err = []
    for indices in seq_indices:
        T_gt_ = np.identity(4)
        T_pred_ = np.identity(4)
        poses_gt = []
        poses_pred = []
        for i in indices:
            T_gt_ = np.matmul(T_gt[i], T_gt_)
            T_pred_ = np.matmul(T_pred[i], T_pred_)
            enforce_orthog(T_gt_)
            enforce_orthog(T_pred_)
            poses_gt.append(T_gt_)
            poses_pred.append(T_pred_)
        err.extend(calcSequenceErrors(poses_gt, poses_pred))
    t_err, r_err = getStats(err)
    return t_err * 100, r_err * 180 / np.pi, err

def saveKittiErrors(err, fname):
    pickle.dump(err, open(fname, 'wb'))

def loadKittiErrors(fname):
    return pickle.load(open(fname, 'rb'))

def save_in_yeti_format(T_gt, T_pred, timestamps, seq_lens, seq_names, root='./'):
    """This function converts outputs to a format that is backwards compatible with the yeti repository."""
    seq_indices = []
    idx = 0
    for s in seq_lens:
        seq_indices.append(list(range(idx, idx + s - 1)))
        idx += (s - 1)

    for s, indices in enumerate(seq_indices):
        fname = root + 'accuracy' + seq_names[s] + '.csv'
        with open(fname, 'w') as f:
            f.write('x,y,yaw,gtx,gty,gtyaw,time1,time2\n')
            for i in indices:
                R_pred = T_pred[i][:3, :3]
                t_pred = T_pred[i][:3, 3:]
                yaw = -1 * np.arcsin(R_pred[0, 1])
                gtyaw = -1 * np.arcsin(T_gt[i][0, 1])
                t = np.matmul(-1 * R_pred.transpose(), np.reshape(t_pred, (3, 1)))
                T = get_inverse_tf(T_gt[i])
                f.write('{},{},{},{},{},{},{},{}\n'.format(t[0, 0], t[1, 0], yaw, T[0, 3], T[1, 3], gtyaw,
                                                           timestamps[i][0], timestamps[i][1]))

def load_icra21_results(results_loc, seq_names, seq_lens):
    T_icra = []
    for i, seq_name in enumerate(seq_names):
        fname = results_loc + 'accuracy' + seq_name + '.csv'
        with open(fname, 'r') as f:
            f.readline()  # Clear out the header
            lines = f.readlines()
            count = 0
            for line in lines:
                line = line.split(',')
                # Retrieve the transform estimated by MC-RANSAC + DOPPLER compensation
                T_icra.append(get_inverse_tf(get_transform(float(line[11]), float(line[12]), float(line[13]))))
                count += 1
            # Append identity transforms at the end in case the ICRA results ended early by a couple frames
            if count < seq_lens[i]:
                print('WARNING: ICRA results shorter than seq_len by {}. Append last TF.'.format((seq_lens[i] - count)))
            while count < seq_lens[i]:
                T_icra.append(T_icra[-1])
                count += 1
    return T_icra

def normalize_coords(coords_2D, width, height):
    """Normalizes coords_2D (BW x N x 2) to be within [-1, 1] """
    batch_size = coords_2D.size(0)
    u_norm = (2 * coords_2D[:, :, 0].reshape(batch_size, -1) / (width - 1)) - 1
    v_norm = (2 * coords_2D[:, :, 1].reshape(batch_size, -1) / (height - 1)) - 1
    return torch.stack([u_norm, v_norm], dim=2)  # BW x num_patches x 2

def convert_to_radar_frame(pixel_coords, config):
    cart_pixel_width = config['cart_pixel_width']
    cart_resolution = config['cart_resolution']
    gpuid = config['gpuid']
    """Converts pixel_coords (B x N x 2) from pixel coordinates to metric coordinates in the radar frame."""
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    B, N, _ = pixel_coords.size()
    R = torch.tensor([[0, -cart_resolution], [cart_resolution, 0]]).expand(B, 2, 2).to(gpuid)
    t = torch.tensor([[cart_min_range], [-cart_min_range]]).expand(B, 2, N).to(gpuid)
    return (torch.bmm(R, pixel_coords.transpose(2, 1)) + t).transpose(2, 1)

def get_indices(batch_size, window_size):
    src_inds = []
    tgt_inds = []
    for i in range(batch_size):
        for j in range(window_size - 1):
            idx = i * window_size + j
            src_inds.append(idx)
            tgt_inds.append(idx + 1)
    return src_inds, tgt_inds

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_T_ba(out, a, b):
    T_b0 = np.eye(4)
    T_b0[:3, :3] = out['R'][0, b].detach().cpu().numpy()
    T_b0[:3, 3:4] = out['t'][0, b].detach().cpu().numpy()
    T_a0 = np.eye(4)
    T_a0[:3, :3] = out['R'][0, a].detach().cpu().numpy()
    T_a0[:3, 3:4] = out['t'][0, a].detach().cpu().numpy()
    return np.matmul(T_b0, get_inverse_tf(T_a0))
