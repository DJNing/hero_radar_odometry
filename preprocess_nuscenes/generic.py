# -*- coding: utf-8 -*-
"""
general operation

"""

from nuscenes import NuScenes
import os.path as osp
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud
import numpy as np
from pyquaternion.quaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import Box

class Generic(object):
    def __init__(self, version, dataset):
        self.dataset = dataset
        self.nusc = NuScenes(version = version, dataroot = dataset, verbose=True)
        self.scene = self.nusc.scene[0]
        self.sample_token = self.scene['first_sample_token']
        self.sample = self.nusc.get('sample', self.sample_token)
        self.channel = 'RADAR_FRONT'
        self.sample_data = self.nusc.get('sample_data', self.sample['data'][self.channel])
        self.anns_token = self.sample['anns']
        self.cs = self.nusc.get('calibrated_sensor', self.sample_data['calibrated_sensor_token'])
        ego_pose_record = self.nusc.get('ego_pose',self.sample_data['ego_pose_token'])
        self.sample_abs_ego_pose = ego_pose_record['translation']

        
        pose_token = self.sample_data['calibrated_sensor_token']
        self.radar_to_car = self.nusc.get('calibrated_sensor', pose_token)
        lidar = self.nusc.get('sample_data', self.sample['data']['LIDAR_TOP'])
        radar = self.nusc.get('sample_data', self.sample['data']['RADAR_FRONT'])
        self.lidar_to_car = self.nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
        self.radar_to_car = self.nusc.get('calibrated_sensor', radar['calibrated_sensor_token'])

        
    def to_scene(self, i):
        self.scene = self.nusc.scene[i]
        self.sample_token = self.scene['first_sample_token']
        self.sample = self.nusc.get('sample', self.sample_token)
    
    def to_next_sample(self):
        self.sample_token = self.sample['next']
        if self.sample_token != '':
            self.sample = self.nusc.get('sample', self.sample_token)

    def get_sample_data(self, channel = 'RADAR_FRONT'):
        self.sample_data = self.nusc.get('sample_data', self.sample['data'][channel])
        return self.sample_data
    
    def get_full_ego_pose(self):
        return self.nusc.get('ego_pose',self.sample_data['ego_pose_token'])

    def get_sample_abs_ego_pose(self, channel = 'RADAR_FRONT'):
        self.sample_data = self.nusc.get('sample_data', self.sample['data'][self.channel])
        ego_pose_record = self.nusc.get('ego_pose',self.sample_data['ego_pose_token'])
        self.sample_abs_ego_pose = ego_pose_record['translation']
        return self.sample_abs_ego_pose

    def get_sample_abs_ego_pose_record(self, channel = 'RADAR_FRONT'):
        self.sample_data = self.nusc.get('sample_data', self.sample['data'][self.channel])
        ego_pose_record = self.nusc.get('ego_pose',self.sample_data['ego_pose_token'])
        return ego_pose_record
    
    def get_pcl(self, channel, sensor='radar'):
        self.sample_data = self.nusc.get('sample_data', self.sample['data'][channel])
        
        pcl_file = osp.join(self.dataset, self.sample_data['filename'])
        if sensor == 'radar':
            pcl = RadarPointCloud.from_file(pcl_file)
        else:
            pcl = LidarPointCloud.from_file(pcl_file)
        pcl.transform(transform_matrix(self.lidar_to_car["translation"], Quaternion(self.lidar_to_car["rotation"])))
        pcl = pcl.points[:4,:].transpose() 
        return pcl
    
    def get_img(self, channel):
        self.sample_data = self.nusc.get('sample_data', self.sample['data'][channel])
        img_file = osp.join(self.dataset, self.sample_data['filename'])
        img = plt.imread(img_file)
        return img

    def is_file_exist(self, channel, sensor='radar'):
        self.sample_data = self.nusc.get('sample_data', self.sample['data'][channel])
        
        pcl_file = osp.join(self.dataset, self.sample_data['filename'])
        return osp.isfile(pcl_file)

    def get_pcl_pano(self, chan = 'RADAR_FRONT', ref_chan = 'RADAR_FRONT'):   
        ref_chan = 'RADAR_FRONT'
        chan = 'RADAR_FRONT'
        chans = ['RADAR_BACK_LEFT','RADAR_BACK_RIGHT', 'RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT']

        pcl_all_ = np.zeros((0,17))
        for chan in chans:
            pc, times = RadarPointCloud.from_file_multisweep(self.nusc, self.sample, chan, ref_chan)
            # transfrom to car coordinate
            pc.transform(transform_matrix(self.radar_to_car["translation"], Quaternion(self.radar_to_car["rotation"])))
            pt = pc.points[:17,:].transpose()
            pcl_all_ = np.concatenate((pcl_all_,pt),axis = 0)   
        
        from sklearn.preprocessing import MinMaxScaler
        rcs = pcl_all_[:,5].reshape(-1,1)
        scaler = MinMaxScaler()
        scaler.fit(rcs)
        rcs = scaler.transform(rcs)
                
        pcl_all_ = np.concatenate((pcl_all_, rcs), axis = 1)

        lidar_pcl = self.get_pcl('LIDAR_TOP', sensor='lidar')
        

        return pcl_all_, lidar_pcl

    def get_box_list(self, channel='radar'):
        
        anns_list = self.sample['anns']
        box_list = []
        if channel == 'radar':
            box_pts = 'num_radar_pts'
        elif channel == 'lidar':
            box_pts = 'num_lidar_pts'
        else:
            raise('Not defined channel: ' + channel)
        for anns in anns_list:
            record = self.nusc.get('sample_annotation', anns)
            radar_pts = record[box_pts]
            if radar_pts > 0:
                box = Box(record['translation'], record['size'], Quaternion(record['rotation']),
                    name=record['category_name'], token=record['token'])
                box_list.append(box)
            else:
                pass

        # project boxes on ego coordinated
        
        return box_list

#    def view(self, pcl):
#        v = pptk.viewer(pcl, point_size=10)
#        v.set(point_size=0.5)