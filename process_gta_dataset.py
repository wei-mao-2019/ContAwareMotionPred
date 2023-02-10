"""
GTA-IM Dataset
"""

import argparse
import os
import pickle
import sys
import time

from tqdm import tqdm

import cv2
import numpy as np
import open3d as o3d

from utils.gta_utils import LIMBS, read_depthmap

sys.path.append('./')


def create_skeleton_viz_data(nskeletons, njoints):
    lines = []
    colors = []
    for i in range(nskeletons):
        cur_lines = np.asarray(LIMBS)
        cur_lines += i * njoints
        lines.append(cur_lines)

        single_color = np.zeros([njoints, 3])
        single_color[:] = [0.0, float(i) / nskeletons, 1.0]
        colors.append(single_color[1:])

    lines = np.concatenate(lines, axis=0)
    colors = np.asarray(colors).reshape(-1, 3)
    return lines, colors


def vis_skeleton_pcd(rec_idx, f_id, fusion_window=20):
    info = pickle.load(open(rec_idx + '/info_frames.pickle', 'rb'))
    info_npz = np.load(rec_idx + '/info_frames.npz')
    info_real = pickle.load(open(rec_idx + '/realtimeinfo.gz', 'rb'))
    room = info_real['setting']['room']

    joints = info_npz['joints_3d_world']
    fn = joints.shape[0]

    splits_idx = np.arange(fn//1000)
    splits = np.array((splits_idx*1000).tolist()+[fn])

    st = time.time()
    # use nearby RGBD frames to create the environment point cloud
    for sp in splits_idx:
        # pcd = o3d.geometry.PointCloud()

        if os.path.exists(f"./data/data_v2_downsample0.02/{rec_idx.split('/')[-1]}_r{room:03d}_sf{sp:d}.npz"):
            continue

        global_pcd = o3d.geometry.PointCloud()
        for i in tqdm(list(range(splits[sp],splits[sp+1],10))):
            fname = rec_idx + '/' + '{:05d}'.format(i) + '.png'
            if os.path.exists(fname):
                infot = info[i]
                cam_near_clip = infot['cam_near_clip']
                if 'cam_far_clip' in infot.keys():
                    cam_far_clip = infot['cam_far_clip']
                else:
                    cam_far_clip = 800.
                depth = read_depthmap(fname, cam_near_clip, cam_far_clip)
                # delete points that are more than 20 meters away
                depth[depth > 10.0] = 0

                # obtain the human mask
                p = info_npz['joints_2d'][i, 0]
                fname = rec_idx + '/' + '{:05d}'.format(i) + '_id.png'
                id_map = cv2.imread(fname, cv2.IMREAD_ANYDEPTH)
                human_id = id_map[
                    np.clip(int(p[1]), 0, 1079), np.clip(int(p[0]), 0, 1919)
                ]

                mask = id_map == human_id
                kernel = np.ones((3, 3), np.uint8)
                mask_dilation = cv2.dilate(
                    mask.astype(np.uint8), kernel, iterations=1
                )
                depth = depth * (1 - mask_dilation[..., None])
                depth = o3d.geometry.Image(depth.astype(np.float32))
                # cv2.imshow('tt', mask.astype(np.uint8)*255)
                # cv2.waitKey(0)

                fname = rec_idx + '/' + '{:05d}'.format(i) + '.jpg'
                color_raw = o3d.io.read_image(fname)

                focal_length = info_npz['intrinsics'][f_id, 0, 0]
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_raw,
                    depth,
                    depth_scale=1.0,
                    depth_trunc=10.0,
                    convert_rgb_to_intensity=False,
                )
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd_image,
                    o3d.camera.PinholeCameraIntrinsic(
                        o3d.camera.PinholeCameraIntrinsic(
                            1920, 1080, focal_length, focal_length, 960.0, 540.0
                        )
                    ),
                )
                depth_pts = np.asarray(pcd.points)

                depth_pts_aug = np.hstack(
                    [depth_pts, np.ones([depth_pts.shape[0], 1])]
                )
                cam_extr_ref = np.linalg.inv(info_npz['world2cam_trans'][i])
                depth_pts = depth_pts_aug.dot(cam_extr_ref)[:, :3]
                pcd.points = o3d.utility.Vector3dVector(depth_pts)
                pcd_down = pcd.voxel_down_sample(voxel_size=0.01)
                global_pcd.points.extend(pcd_down.points)
                global_pcd.colors.extend(pcd_down.colors)
                # if (i//10+1) % 20 == 0:
                #     global_pcd = o3d.geometry.voxel_down_sample(global_pcd, voxel_size=0.005)
                # if (i//10+1) % 100 == 0 and (i//10) < (len(list(range(splits[sp],splits[sp+1],10)))-100):
                #     global_pcd = o3d.geometry.voxel_down_sample(global_pcd, voxel_size=0.01)
        downpcd = global_pcd.voxel_down_sample(voxel_size=0.02)
        points = np.array(downpcd.points,dtype=np.float32)
        colors = (np.array(downpcd.colors)*255).astype(np.uint8)

        np.savez_compressed(f"./data/data_v2_downsample0.02/{rec_idx.split('/')[-1]}_r{room:03d}_sf{sp:d}.npz",joints=joints,scene_points=points,
                            scene_colors=colors,room=room)
    print(f">>> {rec_idx.split('/')[-1]} done, time {time.time()-st:.1f}")
    f = open("./data/files.txt", "a")
    f.writelines([rec_idx.split('/')[-1]+'\n'])
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-pa', '--path', default='./data/GTA-IM-Dataset/')
    parser.add_argument(
        '-f', '--frame', default=180, type=int, help='frame to visualize'
    )
    parser.add_argument(
        '-fw',
        '--fusion-window',
        default=20,
        type=int,
        help='timesteps of RGB frames for fusing',
    )

    list1 = []

    args = parser.parse_args()
    for file in tqdm(os.listdir(args.path)):
        if '2020' not in file:
            continue
        vis_skeleton_pcd(args.path + '/' + file, args.frame, args.fusion_window)
