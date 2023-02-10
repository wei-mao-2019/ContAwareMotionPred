import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

class DatasetGTA(Dataset):

    def __init__(self, mode, dataset_specs):
        self.mode = mode
        self.t_his = t_his = dataset_specs.get('t_his',30)
        self.t_pred = t_pred = dataset_specs.get('t_pred',60)
        self.t_total = t_his + t_pred
        self.random_rot = dataset_specs.get('random_rot',False)
        self.is_contact = dataset_specs.get('is_contact',False)
        self.is_frame_contact = dataset_specs.get('is_frame_contact',False)
        self.step = step = dataset_specs.get('step',1)
        self.num_scene_points = num_scene_points = dataset_specs.get('num_scene_points', 10000)
        self.max_dist_from_human = max_dist_from_human = dataset_specs.get('max_dist_from_human', 2.5)
        self.wscene = wscene = dataset_specs.get('wscene',True)
        self.wcont = wcont = dataset_specs.get('wcont',True)
        self.num_cont_points = num_cont_points = dataset_specs.get('num_cont_points', 500)
        self.sigma = dataset_specs.get('sigma', 0.02)
        self.cont_thre = 0.2

        self.data_file = data_file = dataset_specs.get('data_file', '/home/wei/Documents/projects/2021-human-scene-intraction/scene-aware-motion-prediction/data/GTA')
        self.scene_split = {'train': ['r001','r002','r003', 'r006'],

                            #'test': ['r010', 'r011', 'r013'],
                            'test': ['r010', 'r003', 'r013']
                            }
        self.pose = {}
        self.scene = {}
        self.idx2scene = {}

        print('read original data file')
        for i, seq in tqdm(enumerate(os.listdir(data_file))):
            # if '2020-06-04-22-57-20_r013_sf0' not in seq:
            #     continue
            room = seq.split('_')[1]
            if room not in self.scene_split[mode]:
                continue
            data_tmp=np.load(f'{data_file}/{seq}',allow_pickle=True)
            self.pose[i] = data_tmp['joints']
            self.idx2scene[i] = seq[:-4]
            # if wscene or wcont:
            self.scene[i] = data_tmp['scene_points']

            ########### for debug
            if len(self.pose) > 0:
                break

        self.data = {}
        self.scene_point_idx = {}
        self.cont_idx = {}
        self.sdf_coord_idxs = {}
        k = 0
        min_num_scene = 1000000
        max_num_scene = 0
        print("generateing data idxs")
        for sub in tqdm(self.pose.keys()):
            room = self.idx2scene[sub].split('_')[1]
            seq_len = self.pose[sub].shape[0]
            idxs_frame = np.arange(0,seq_len - self.t_total + 1,step)
            for i in idxs_frame:
                self.data[k] = f'{sub}.{i}'

                pose = self.pose[sub][i:self.t_total+i]
                # root joints is spline_4 index 14
                root_joint = pose[t_his-1, 14:15]
                scene_vert = self.scene[sub]
                dist = np.linalg.norm(scene_vert - root_joint, axis=-1)
                idxs = np.where(dist <= self.max_dist_from_human)[0]
                self.scene_point_idx[k] = idxs.astype(np.int32)

                if min_num_scene > len(idxs):
                    min_num_scene = len(idxs)

                if max_num_scene < len(idxs):
                    max_num_scene = len(idxs)

                k += 1

        print(f"num of scene points from {min_num_scene:d} to {max_num_scene:d}")
        print(f'seq length {self.t_total},in total {k} seqs')

    def __len__(self):
        return len(list(self.data.keys()))


    def __getitem__(self, idx):

        item_key = self.data[idx].split('.')
        sub = int(item_key[0])
        fidx = int(item_key[1])
        subj = self.idx2scene[sub]
        room = subj.split('_')[1]
        item_key = f"{subj}.{item_key[1]}"

        pose = torch.tensor(self.pose[sub][fidx:fidx + self.t_total]).float()
        scene_origin = torch.clone(pose[self.t_his-1,14:15])

        scene_vert = self.scene[sub]# data_tmp['scene_points']
        scene_vert = torch.tensor(scene_vert).float()

        idxs = self.scene_point_idx[idx]
        len_vidx = len(idxs)
        if len_vidx < self.num_scene_points:
            ids = list(range(len_vidx)) + np.random.choice(np.arange(len_vidx),self.num_scene_points-len_vidx).tolist()
        else:
            ids = np.random.choice(np.arange(len_vidx), self.num_scene_points, replace=False)
        v_idx = idxs[ids]
        scene_vert = scene_vert[v_idx]
        scene_vert = scene_vert - scene_origin # [5000, 3]
        pose = pose - scene_origin # [90, 21, 3]

        return pose, scene_vert, scene_origin, 0, item_key

if __name__ == '__main__':
    dataset = DatasetGTA('train',{})
    samp = dataset.sample()
    loader = dataset.sampling_generator(num_samples=1000, batch_size=8)
    for scene_vert,scene_norm,scene_cont,betas,global_expmap,\
        is_female,full_pose,global_transl,expression in loader:
        print(11)
    print(11)
