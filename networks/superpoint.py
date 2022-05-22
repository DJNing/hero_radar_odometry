import torch
import torch.nn.functional as F
import torch.nn as nn
try: from networks.unet_parts import *
except: from unet_parts import *
from networks.layers import DoubleConv, OutConv, Down, Up

class SuperpointEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65
        self.inc = inconv(1, c1)
        self.down1 = down(c1, c2)
        self.down2 = down(c2, c3)
        self.down3 = down(c3, c4)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x4


class DetectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 64
        is_resize = True
        self.relu = nn.ReLU(inplace=True)
        self.conv0 = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(c5)
        self.conv1 = nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(det_h)

    def forward(self, x):
        if type(x) == list:
            ip_feat = x[-1]
        else:
            ip_feat = x
        op0 = self.relu(self.bn0(self.conv0(ip_feat)))
        op1 = self.bn1(self.conv1(op0))
        return op1


class DescriptorHead(nn.Module):
    def __init__(self):
        super().__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65
        self.relu = nn.ReLU(inplace=True)
        self.conv0 = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(c5)
        self.conv1 = nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(d1)

    def forward(self, x):
        if type(x) == list:
            ip_feat = x[-1]
        else:
            ip_feat = x
        op0 = self.relu(self.bn0(self.conv0(ip_feat)))
        op1 = self.bn1(self.conv1(op0))
        return op1
        

class ScoreHead(nn.Module):
    def __init__(self):
        super().__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 64
        self.relu = nn.ReLU(inplace=True)
        self.conv0 = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(c5)
        self.conv1 = nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(det_h)

    def forward(self, x):
        if type(x) == list:
            ip_feat = x[-1]
        else:
            ip_feat = x
        op0 = self.relu(self.bn0(self.conv0(ip_feat)))
        op1 = self.bn1(self.conv1(op0))
        return op1 # N x 64 x H/8 x W/8


class SuperPoint_modified(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SuperpointEncoder()
        self.det_head = DetectionHead()
        self.score_head = ScoreHead()
        self.desc_head = DescriptorHead()

    def forward(self, data):
        # data = batch['data'].to(self.gpuid)
        _, _, height, width = data.size()
        encoded_feat = self.encoder(data)
        det = self.det_head(encoded_feat) # B, 64, H/8, W/8
        desc = self.desc_head(encoded_feat)
        score = self.score_head(encoded_feat)
        _, _, H, W = det.size()
        det = det.view([-1, 1, 8*H, 8*W]) # B, 1, H, W
        score = score.view([-1, 1, 8*H, 8*W]) # B, 1, H, W

        # make sure the recover the origianl size
        det = F.interpolate(det, size=(height, width), mode='bilinear')
        score = F.interpolate(score, size=(height, width), mode='bilinear')
        desc = F.interpolate(desc, size=(height, width), mode='bilinear') # B, 256, H, W

        return det, score, desc

    def load_origianal_pretrain(self, fname):
        pass
        

class SuperPoint_original(nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self, config):
        super(SuperPoint_original, self).__init__()
        bilinear = config['networks']['unet']['bilinear']
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 64
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        # Score Head
        self.score_head = ScoreHead()

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
            x: Image pytorch tensor shaped N x 1 x H x W.
        Output
            semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
            desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        _, _, height, width = x.size()
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        det = self.convPb(cPa)
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        score = self.score_head(x)
        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.

        # post processing to adapt to UTR
        _, _, H, W = det.size()
        det = det.view([-1, 1, 8*H, 8*W]) # B, 1, H, W
        score = score.view([-1, 1, 8*H, 8*W]) # B, 1, H, W

        # make sure the recover the origianl size
        det = F.interpolate(det, size=(height, width), mode='bilinear')
        score = F.interpolate(score, size=(height, width), mode='bilinear')
        desc = F.interpolate(desc, size=(height, width), mode='bilinear') # B, 256, H, W

        return det, score, desc


class SuperPoint_UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        bilinear = config['networks']['unet']['bilinear']
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        o1, o2, o3, o4 = 4, 8, 16, 32
        det_h = 64
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        self.conv5a = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        # Detector Head.
                
        self.det_up3 = up((c4 + c5), o4, bilinear)
        self.det_up2 = up((c3 + o4), o3, bilinear)
        self.det_up1 = up((c2 + o3), o2, bilinear)
        # self.det_up1 = up((c2 + o3), o1, bilinear)
        self.det_pts = OutConv(o2, 1)

        # Descriptor Head.
        # self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        # self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        

        # Score Head
        self.scr_up3 = up((c4 + c5), o4, bilinear)
        self.scr_up2 = up((c3 + o4), o3, bilinear)
        self.scr_up1 = up((c2 + o3), o2, bilinear)
        # self.scr_up1 = up((c2 + o3), o1, bilinear)
        self.score_out = OutConv(o2, 1)

    def forward(self, x):
        # Shared Encoder.
        _, _, height, width = x.size()
        x0 = self.relu(self.conv1a(x))
        x0 = self.relu(self.conv1b(x0)) # C = C1
        x1 = self.pool(x0)
        x1 = self.relu(self.conv2a(x1))
        x1 = self.relu(self.conv2b(x1)) # C = C2
        x2 = self.pool(x1) 
        x2 = self.relu(self.conv3a(x2))
        x2 = self.relu(self.conv3b(x2)) # C = C3
        x3 = self.pool(x2) 
        x3 = self.relu(self.conv4a(x3))
        x3 = self.relu(self.conv4b(x3)) # C = C4
        x4 = self.pool(x3)
        x4 = self.relu(self.conv5a(x3)) # C = C5
        # Detector Head.
        x3_det_up = self.det_up3(x4, x3)
        x2_det_up = self.det_up2(x3_det_up, x2)
        x1_det_up = self.det_up1(x2_det_up, x1) # B, _, H, W
        det = self.det_pts(x1_det_up)

        # Score Head
        x3_score_up = self.scr_up3(x4, x3)
        x2_score_up = self.scr_up2(x3_score_up, x2)
        x1_score_up = self.scr_up1(x2_score_up, x1) # B, _, H, W
        score = self.score_out(x1_score_up)

        # Descriptor Head.
        # cDa = self.relu(self.convDa(x))
        # desc = self.convDb(cDa)
        # score = self.score_head(x)
        # dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        # desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.

        # post processing to adapt to UTR
        # _, _, H, W = det.size()
        # det = det.view([-1, 1, 8*H, 8*W]) # B, 1, H, W
        # score = score.view([-1, 1, 8*H, 8*W]) # B, 1, H, W

        # make sure the recover the origianl size
        det = F.interpolate(det, size=(height, width), mode='bilinear')
        score = F.interpolate(score, size=(height, width), mode='bilinear')
        # desc = F.interpolate(desc, size=(height, width), mode='bilinear') # B, 256, H, W
        f1 = torch.cat([x1_det_up, x1_score_up], dim=1)
        f1 = F.interpolate(f1, size=(height, width), mode='bilinear') # 64
        f2 = torch.cat([x2_det_up, x2_score_up], dim=1)
        f2 = F.interpolate(f2, size=(height, width), mode='bilinear') # 128
        f3 = torch.cat([x3_det_up, x3_score_up], dim=1)
        f3 = F.interpolate(f3, size=(height, width), mode='bilinear') # 128
        desc = torch.cat([f1, f2, f3], dim=1)

        return det, score, desc

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SuperPoint_original()
    model = model.to(device)

    # check keras-like model summary using torchsummary
    from torchsummary import summary
    summary(model, input_size=(1, 480, 480))
    ckpt_file = '/workspace/hero_radar_odometry/pretrain_spp/superpoint_v1.pth'
    ckpt_dict = torch.load(ckpt_file) 
    model_dict = model.state_dict()
    rand_ip = torch.randn([1, 1, 480, 480]).to(device)
    det, desc, score, x = model(rand_ip)
    score_head = ScoreHead().to(device)

