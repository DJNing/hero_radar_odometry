import numpy as np
import os
import csv
import shutil
import argparse
import tqdm

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--save_dir', type=str, default='/home/dj/data/preprocess_nuScenes/test',
                    help='root dir for saving data')
parser.add_argument('--source_img_dir', type=str, default='/home/dj/data/MVDNet/data/RobotCar/object/lidar',
                    help='image source dir')
parser.add_argument('--scene_pose_dict', type=str, default='/home/dj/data/preprocess_nuScenes/scene_pose.npy',
                    help='saved file for pose')



if __name__ == "__main__":
    args = parser.parse_args()
    save_dir = args.save_dir
    gt_head = ['source_timestamp', 'destination_timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'source_radar_timestamp', 'destination_radar_timestamp']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    img_src_dir = args.source_img_dir
    tag = img_src_dir.split('/')[-1]

    save_dir = os.path.join(save_dir, tag)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # load seq_pose_dict
    seq_pose_dict_file = args.scene_pose_dict
    seq_pose_dict = np.load(seq_pose_dict_file, allow_pickle=True).item()
    # print(len(seq_pose_dict))


    # ====================================================
    # construct data
    # ====================================================

    for seq in tqdm.tqdm(seq_pose_dict):
        # gather timestamp
        # the first image of each scene is not available
        try:
            pose_list = seq_pose_dict[seq]
            init = 0
            cnt = 0

            temp_dir = os.path.join(save_dir, seq)
            if not os.path.exists(temp_dir):
                os.mkdir(temp_dir)
            gt_dir = os.path.join(temp_dir, 'gt')
            if not os.path.exists(gt_dir):
                os.mkdir(gt_dir)
            radar_dir = os.path.join(temp_dir, 'radar')
            if not os.path.exists(radar_dir):
                os.mkdir(radar_dir)
            ts_file = os.path.join(temp_dir, 'radar.timestamps')
            gt_file = os.path.join(gt_dir, 'radar_odometry.csv')

            ts_list = []
            gt_list = []
            pair_dict = {}
            gt_list.append(gt_head)
            for pose in pose_list:
                src_ts = pose['src']
                dst_ts = pose['dst']
                dst_cnt = pose['dst_cnt']
                # print(src_cnt)
                # print(src_ts)
                if init == 0:
                    init = 1
                    # remove the first image of each scene, no gt here
                else:
                    gt_temp = [src_ts, dst_ts, pose['x'], pose['y'], pose['z'], pose['roll'], pose['pitch'], pose['yaw'], src_ts, dst_ts]
                    gt_list.append(gt_temp)
                    ts_list.append(dst_ts)  
                    pair_dict[str(dst_ts)] = str(dst_cnt)
                
            # gt_dict[seq] = gt_list
            # ts_dict[seq] = ts_list
            with open(ts_file, 'w') as f:
                writer = csv.writer(f, delimiter=' ')
                for ts in ts_list:
                    ox_ts = [str(ts), str(1)]
                    writer.writerow(ox_ts)
            with open(gt_file, 'w') as f:
                writer = csv.writer(f, delimiter=',')
                for gt in gt_list:
                    writer.writerow(gt)
            
            # move file to the corresponding folder 
            # print(pair_dict)
            # break
            dst_img_dir = os.path.join(temp_dir, 'radar')
            for key in pair_dict:
                f_cnt = int(pair_dict[key])
                fname = '{:0>5d}.png'.format(f_cnt)
                # print(fname)
                src_file = os.path.join(img_src_dir, fname)
                dst_file = os.path.join(dst_img_dir, key+'.png')
                shutil.copy(src_file, dst_file)
        except FileNotFoundError:
            shutil.rmtree(temp_dir)
            break

