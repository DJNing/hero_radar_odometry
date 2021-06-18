from generic import Generic
from matplotlib import pyplot as plt
import tqdm
import numpy as np
import scipy.io as io
import cv2
import argparse
import os
from scipy import signal
import math
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
import csv
from scipy.spatial.transform import Rotation as R

CONFIG = {
    "minX": -60,
    "maxX": 60,
    "minY": -60,
    "maxY": 60,
    "minZ": 0.5,
    "maxZ": 3,
    "res":.25,
    "save_dir":"/home/dj/data/MVDNet/data/RobotCar/object",
    "pass_cnt":1000,
    "save_cnt":100,
    "width":512,
    "p_width":64
}
pix_range = int(CONFIG['width']/2)
sensor_range = int(pix_range*CONFIG['res'])
CONFIG['minX'] = -sensor_range
CONFIG['maxX'] = sensor_range
CONFIG['minY'] = -sensor_range
CONFIG['maxY'] = sensor_range
p_channel = CONFIG['width']/CONFIG['p_width']
p_shape = [1, p_channel*p_channel, CONFIG['p_width'], CONFIG['p_width']]


# check save dir 
save_dir = CONFIG['save_dir']
lidar_dir = os.path.join(save_dir, 'lidar')
radar_dir = os.path.join(save_dir, 'radar')
front_cam_dir = os.path.join(save_dir, 'cam_front')

dir_list = [save_dir, lidar_dir, radar_dir, front_cam_dir]

for i in dir_list:
    if os.path.exists(i):
        pass
    else:
        os.mkdir(i)    
    
# if os.path.exists(overlaid_dir):
#     pass
# else:
#     os.mkdir(overlaid_dir)

import math
 
def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians


def removePoints2D(PointCloud, BoundaryCond):
    # Boundary condition
    minX = BoundaryCond['minX']
    maxX = BoundaryCond['maxX']
    minY = BoundaryCond['minY']
    maxY = BoundaryCond['maxY']
  
    # Remove the point out of range x,y,z
    mask = np.where(
            (PointCloud[:, 0] >= minX) &
            (PointCloud[:, 0] <= maxX) &
            (PointCloud[:, 1] >= minY) &
            (PointCloud[:, 1] <= maxY) 
            )
    PointCloud = PointCloud[mask]
    return PointCloud

def removePoints(PointCloud, BoundaryCond):
    # Boundary condition
    minX = BoundaryCond['minX']
    maxX = BoundaryCond['maxX']
    minY = BoundaryCond['minY']
    maxY = BoundaryCond['maxY']
    minZ = BoundaryCond['minZ']
    maxZ = BoundaryCond['maxZ']
  
    # Remove the point out of range x,y,z
    mask = np.where(
            (PointCloud[:, 0] >= minX) &
            (PointCloud[:, 0] <= maxX) &
            (PointCloud[:, 1] >= minY) &
            (PointCloud[:, 1] <= maxY) &
            (PointCloud[:, 2] >= minZ) &
            (PointCloud[:, 2] <= maxZ)
            )
    PointCloud = PointCloud[mask]
    PointCloud[:, 2] = PointCloud[:, 2] - minZ
    return PointCloud

def drawBEV(pcds, size, res=.125):
    # center coord
    x, y = size
    cx = int(x/2)
    cy = int(y/2)
    image = np.zeros([x,y,3]) 
    for i in pcds:
        ix = i[0]/res + cx
        iy = i[1]/res + cy
        ix = int(ix)
        iy = int(iy)
        image[ix, iy,:] = 1
    return image


def project_box(bbx, pose_record):
    box_list = []
    for box in bbx:
        yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        box_list.append(box)
    return box_list

def box_to_frame(box, res, radius, mode='pix'):
    if mode == 'pix':
        c_raw = box.center[:2]
        wl = box.wlh[:2]
        c = c_raw/CONFIG['res'] + radius
        wl = wl/CONFIG['res']
        min_c = min(c_raw)
        max_c = max(c_raw)
        if (min_c < -60/CONFIG['res']) or (max_c > 60):
            return None
    elif mode == 'meter':
        c = box.center[:2]
        wl = box.wlh[:2]
        min_c = min(c)
        max_c = max(c)
        if (min_c < -60) or (max_c > 60):
            return None
    
    q = box.orientation
    yaw, pitch, roll = q.yaw_pitch_roll
    yaw = 180 * np.array([yaw])/math.pi
    box_coord = np.hstack((c, wl, yaw))
    return box_coord

def render_box(img, box, color=(0,1,0), thickness=1):
    box_corner = box_to_frame(box, CONFIG['res'], 255)
    for i in range(box_corner.shape[1]):
        if i == 3:
            start_point = box_corner[:, i]
            end_point = box_corner[:, 0]
        else:
            start_point = box_corner[:, i]
            end_point = box_corner[:, i+1]
        img = cv2.line(img, tuple([start_point[1], start_point[0]]), tuple([end_point[1], end_point[0]]), tuple(color), thickness=thickness)
    return img

def render_boxes(img, boxes, color=(0,1,0), thickness=1):
    for box in boxes:
        if 'vehicle' in box.name:
            img = render_box(img, box, color, thickness)
    return img

def dump_box(anns, file_name):
    with open(file_name, mode='w') as f:
        writer = csv.writer(f, delimiter=' ')
        cnt = 0
        for i in anns:
            box_list = box_to_list(i)
            box_list = ['car', str(cnt)] + box_list
            writer.writerow(box_list)
            cnt += 1

def box_to_list(box):
    result = []
    for i in box:
        result.append(str(i))
    return result



def state_dict(tr0, tr1, src_ts, dst_ts, dst_cnt):
    # reverse
    rel = np.linalg.inv(np.dot(np.linalg.inv(tr0), tr1))
    rot = rel[:3,:3]
    r = R.from_matrix(rot)
    roll, pitch, yaw = r.as_euler('XYZ')
    x,y,z = rel[:3,3]
    temp = {#'src':src_ts,
            #'dst':dst_ts,
            #'dst_cnt':dst_cnt,
            #'src_cnt':dst_cnt-1,
            ## switch dst and src
            'dst':src_ts,
            'src':dst_ts,
            'src_cnt':dst_cnt,
            'dst_cnt':dst_cnt-1,
            'x':'{:.6f}'.format(float(x)),
            'y':'{:.6f}'.format(float(y)),
            'z':'{:.6f}'.format(float(z)),
            'yaw':'{:.6f}'.format(float(yaw)),
            'pitch':'{:.6f}'.format(float(pitch)),
            'roll':'{:.6f}'.format(float(roll))}
    return temp

data_root = "/home/dj/temp"
version = "v1.0-trainval"
# save_root = './test'
# if not os.path.exists(save_root):
#     os.mkdir(save_root)

val = Generic('v1.0-trainval', data_root)

seq_pose_dict = {}

cnt = 0
max_cnt = 10119
for i in tqdm.tqdm(range(len(val.nusc.scene))):
    val.to_scene(i)
    scene_key = 'nuScenes_'+str(i)
    pose_list = []
    nbr_samples = val.scene['nbr_samples']
    tr0 = None
    # print(nbr_samples)
    for j in range(nbr_samples):
        if not val.is_file_exist('LIDAR_TOP'):
            continue
        ts = val.sample_data['timestamp']
        pose = val.get_full_ego_pose()
        if tr0 is None:
            tr0 = transform_matrix(pose['translation'], Quaternion(pose['rotation']))
            src_ts = pose['timestamp']
        else:
            tr1 = transform_matrix(pose['translation'], Quaternion(pose['rotation']))
            dst_ts = pose['timestamp']
            temp = state_dict(tr0, tr1, src_ts, dst_ts, cnt)
            # print(cnt)
            pose_list.append(temp)
            tr0 = tr1
            src_ts = dst_ts        
        
        val.to_next_sample()
        cnt += 1
        # if cnt > max_cnt:
        #     exit
        # break
    if len(pose_list) != 0:
        seq_pose_dict[scene_key] = pose_list

print('saveing file with total ' + str(len(seq_pose_dict)) +' scenes')
np.save('scene_pose.npy', seq_pose_dict)