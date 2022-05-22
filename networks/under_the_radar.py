import torch
from networks.unet import UNet
from networks.keypoint import Keypoint
from networks.softmax_matcher import SoftmaxMatcher
from networks.svd import SVD
import torch.nn as nn
from networks.superpoint import SuperPoint_original, SuperPoint_UNet

class UnderTheRadar(torch.nn.Module):
    """
        This model computes a 3x3 Rotation matrix and a 3x1 translation vector describing the transformation
        between two radar scans. This transformation can be used for odometry or metric localization.
        It is intended to be an implementation of Under the Radar (Barnes and Posner, 2020)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gpuid = config['gpuid']
        self.unet = UNet(config)
        self.keypoint = Keypoint(config)
        self.softmax_matcher = SoftmaxMatcher(config)
        self.svd = SVD(config)

    def forward(self, batch):
        data = batch['data'].to(self.gpuid)

        detector_scores, weight_scores, desc = self.unet(data)

        keypoint_coords, keypoint_scores, keypoint_desc = self.keypoint(detector_scores, weight_scores, desc)

        pseudo_coords, match_weights, kp_inds = self.softmax_matcher(keypoint_scores, keypoint_desc, weight_scores, desc)
        src_coords = keypoint_coords[kp_inds]

        R_tgt_src_pred, t_tgt_src_pred = self.svd(src_coords, pseudo_coords, match_weights)

        return {'R': R_tgt_src_pred, 't': t_tgt_src_pred, 'scores': weight_scores,
                'src': src_coords, 'tgt': pseudo_coords, 'match_weights': match_weights, 'dense_weights': weight_scores}

class UnderTheSuperPoint(nn.Module):
    """
        This model computes a 3x3 Rotation matrix and a 3x1 translation vector describing the transformation
        between two radar scans. This transformation can be used for odometry or metric localization.
        It is intended to be an implementation of Under the Radar (Barnes and Posner, 2020)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gpuid = config['gpuid']
        if config['backbone'] == "SuperPoint_original":
            self.unet = SuperPoint_original()
        elif config['backbone'] == "SuperPoint_UNet":
            self.unet = SuperPoint_UNet(config)
        else:
            raise NotImplementedError()
        self.keypoint = Keypoint(config)
        self.softmax_matcher = SoftmaxMatcher(config)
        self.svd = SVD(config)

    def forward(self, batch):
        data = batch['data'].to(self.gpuid)

        detector_scores, weight_scores, desc = self.unet(data)

        keypoint_coords, keypoint_scores, keypoint_desc = self.keypoint(detector_scores, weight_scores, desc)

        pseudo_coords, match_weights, kp_inds = self.softmax_matcher(keypoint_scores, keypoint_desc, weight_scores, desc)
        src_coords = keypoint_coords[kp_inds]

        R_tgt_src_pred, t_tgt_src_pred = self.svd(src_coords, pseudo_coords, match_weights)

        return {'R': R_tgt_src_pred, 't': t_tgt_src_pred, 'scores': weight_scores,
                'src': src_coords, 'tgt': pseudo_coords, 'match_weights': match_weights, 'dense_weights': weight_scores}

    def load_backbone_ckpt(self, fname):
        ckpt_dict = torch.load(fname)
        backbone_dict = self.unet.state_dict()
        from copy import deepcopy as dc
        temp_dict = dc(backbone_dict)
        loaded_key = []
        for key in backbone_dict.keys():
            if ckpt_dict.get(key, False):
                if backbone_dict[key].size() == ckpt_dict[key].size():
                    temp_dict[key] = ckpt_dict[key]
                    loaded_key.append(key)
        self.unet.load_state_dict(temp_dict)
        print('loaded keys: ', loaded_key)


if __name__ == '__main__':
    import json
    cfg_file = '/workspace/hero_radar_odometry/config/medical.json'
    with open(cfg_file) as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UnderTheSuperPoint(config)
    model = model.to(device)

    # check keras-like model summary using torchsummary
    from torchsummary import summary
    # summary(model, input_size=(1, 480, 480))
    ckpt_file = '/workspace/hero_radar_odometry/pretrain_spp/superpoint_v1.pth'
    ckpt_dict = torch.load(ckpt_file) 
    model_dict = model.state_dict()
    from copy import deepcopy as dc
    temp_dict = dc(model_dict)
    for key in model_dict.keys():
        temp_key = key[5:]
        if ckpt_dict.get(temp_key, False):
            if temp_dict[temp_key].size() == ckpt_dict[temp_key].size():
                temp_dict[temp_key] = ckpt_dict[temp_key]
    


