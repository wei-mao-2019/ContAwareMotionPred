import os
import sys
import math
import argparse
import time
import cv2
import open3d as o3d
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.getcwd())
from utils.config import Config
from datasets.dataset_gta import DatasetGTA
from models.motion_pred import *
from utils import *
from utils.util import *
from utils.gta_utils import LIMBS

def create_skeleton_viz_data(nskeletons, njoints, col):
    lines = []
    colors = []
    for i in range(nskeletons):
        cur_lines = np.asarray(LIMBS)
        cur_lines += i * njoints
        lines.append(cur_lines)

        single_color = np.zeros([njoints, 3])
        single_color[:] = col
        colors.append(single_color[1:])

    lines = np.concatenate(lines, axis=0)
    colors = np.asarray(colors).reshape(-1, 3)
    return lines, colors

def add_skeleton(vis, joints, col, line_set=None,sphere_list=None,jlast=None):
    # add gt
    tl, jn, _ = joints.shape
    joints = joints.reshape(-1, 3)
    if jlast is not None:
        jlast = jlast.reshape(-1, 3)

    # plot history
    nskeletons = tl
    lines, colors = create_skeleton_viz_data(nskeletons, jn, col=col)
    if line_set is None:
        line_set = o3d.geometry.LineSet()
        vis.add_geometry(line_set)

    line_set.points = o3d.utility.Vector3dVector(joints)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis.update_geometry(line_set)

    count = 0
    sphere_list_tmp = []
    for j in range(joints.shape[0]):
        # spine joints
        if j % jn == 11 or j % jn == 12 or j % jn == 13:
            continue
        transformation = np.identity(4)
        if jlast is not None:
            transformation[:3, 3] = joints[j]-jlast[j]
        else:
            transformation[:3, 3] = joints[j]
        # head joint
        if j % jn == 0:
            r = 0.07
        else:
            r = 0.03
        if sphere_list is None:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)
            vis.add_geometry(sphere)
            sphere_list_tmp.append(sphere)
        else:
            sphere = sphere_list[count]
        sphere.paint_uniform_color(col)
        sphere.transform(transformation)
        vis.update_geometry(sphere)
        count += 1
    if sphere_list is None:
        sphere_list = sphere_list_tmp
    return line_set,sphere_list

@torch.no_grad()
def train(epoch):
    thres = math.exp(-0.5 * args.cont_thre ** 2 / dataset.sigma ** 2)
    root_joint_idx = 14
    root_idx = [root_joint_idx*3,root_joint_idx*3+1,root_joint_idx*3+2]

    generator = DataLoader(dataset,batch_size=cfg.batch_size,shuffle=True,
                           num_workers=2,pin_memory=True,drop_last=False)

    pad_idx = list(range(t_his)) + [t_his-1]*t_pred

    y_for_save = {}
    for pose, scene_vert, scene_origin, _, item_key in tqdm(generator):

        bs = pose.shape[0]
        nj = pose.shape[2]
        scene_vert = scene_vert.to(device=device) # [:,:10000]
        npts = scene_vert.shape[1]

        joints = pose.to(device=device)

        is_cont = (scene_vert[:, None, :, None, :] - joints[:, :, None, :, :]).norm(dim=-1)
        is_cont_gauss = torch.exp(-0.5*is_cont**2/dataset.sigma**2)

        joints_orig = joints[:, :, 14:15]
        joints = joints - joints_orig
        joints[:, :, 14:15] = joints_orig

        if args.w_est_cont:
            is_cont_pad = is_cont_gauss[:, pad_idx].reshape([bs, t_his + t_pred, -1])
            is_cont_dct = torch.matmul(dct_m_cont[None], is_cont_pad).reshape([bs, dct_n_cont, npts, nj])
            is_cont_dct = is_cont_dct.permute(0, 1, 3, 2).reshape([bs, dct_n_cont * nj, npts])

            # def forward(self, x, scene, aux=None, cont_dct=None):
            cont_dct_est = model_cont(joints[:, :t_his].reshape([bs, t_his, -1]).transpose(0, 1), scene_vert.transpose(1, 2),
                                      cont_dct=is_cont_dct)  # (x, z, scene, aux_data=None, horizon=30, nk=5)
            cont_dct_est = cont_dct_est.reshape([bs, dct_n_cont, nj, npts]).reshape([bs, dct_n_cont, nj * npts])
            cont_est = torch.matmul(idct_m_cont[None], cont_dct_est)
            cont_est = cont_est.reshape([bs, t_his + t_pred, nj, npts]).transpose(2, 3)[:,t_his:]
            is_cont_est = 1- cont_est

            min_dist_value = (is_cont_est.min(dim=2)[0] < (1-thres)).to(dtype=dtype)
            min_dist_idx = is_cont_est.min(dim=2)[1].reshape([-1])
            idx_tmp = torch.arange(bs, device=device)[:, None].repeat([1, t_pred * nj]).reshape([-1])

            cont_points = scene_vert[idx_tmp, min_dist_idx, :].reshape([bs, t_pred, nj, 3])
            cont_points = cont_points * min_dist_value[..., None]
            cont_points = torch.cat([cont_points, min_dist_value[..., None]], dim=-1)

        if not args.w_est_cont:
            dist = is_cont
            min_dist_value = (dist.min(dim=2)[0] < 0.3).to(dtype=dtype)
            min_dist_idx = dist.min(dim=2)[1].reshape([-1])
            idx_tmp = torch.arange(bs, device=device)[:, None].repeat([1, (t_pred + t_his) * nj]).reshape([-1])

            cont_points = scene_vert[idx_tmp, min_dist_idx, :].reshape([bs, t_his + t_pred, nj, 3])
            cont_points = cont_points * min_dist_value[..., None]
            cont_points = torch.cat([cont_points, min_dist_value[..., None]], dim=-1)[:,t_his:]

        # def forward(self, x, cont, cont_mask, aux=None, horizon=30,
        #             dct_m=None, idct_m=None, root_idx=None):
        y, root_traj = model(joints[:,:t_his].reshape([bs,t_his,-1]).transpose(0,1),
                              cont_points.reshape([bs,t_pred,-1]) if wcont else None,
                              None,
                              None, t_pred,
                              dct_m=dct_m, idct_m=idct_m,
                              root_idx=root_idx)
        y[:,:,root_idx] = root_traj[:,t_his:].transpose(0,1)
        y = y.transpose(0, 1)
        y = y.reshape([bs, t_pred, -1, 3])

        is_cont = is_cont_gauss.cpu().data.numpy()
        print(is_cont.shape)
        cont_est = cont_est.cpu().data.numpy()
        joints = joints.cpu().data.numpy()
        root_joint = joints[:,:,root_joint_idx:root_joint_idx+1]
        joints = joints + root_joint
        joints[:,:,root_joint_idx:root_joint_idx+1] = root_joint
        y = y.cpu().data.numpy()
        root_joint = y[:,:,root_joint_idx:root_joint_idx+1]
        y = y + root_joint
        y[:,:,root_joint_idx:root_joint_idx+1] = root_joint
        print(y.shape)

        """visualization"""
        idxs_vis = np.arange(0, 90, 1)
        his_col = [0.5, 0.5, 0.5]
        cont_col = [[1.0, 0.0, 0.0]]
        cont_col_est = [[0.0, 0.0, 1.0]]
        furt_col = [1.0, 0.5, 0.5]
        est_col = [[0.5, 1.0, 0.5], [1.0, 1.0, 0.5], [0.5, 0.5, 1.0], [1.0, 0.5, 1.0], [0.5, 1.0, 1.0]]

        vis = o3d.visualization.Visualizer()
        vis.create_window(left=0, top=0, window_name='motion')
        cv2.imshow('frame', np.zeros([100, 100, 3]))

        for bs_idx in range(bs):
            sn = item_key[bs_idx].split('.')[0]
            print(scene_vert.shape)
            scene_point = scene_vert[bs_idx].cpu().data.numpy()
            scene_color = np.ones_like(scene_point)*0.8

            scene_origin_new = np.mean(scene_point, keepdims=True, axis=0)
            scene_point = scene_point - scene_origin_new
            joints[bs_idx] = joints[bs_idx] - scene_origin_new[None, ...]
            y[bs_idx] = y[bs_idx] - scene_origin_new[None, None, ...]

            scene = o3d.geometry.PointCloud()
            scene.points = o3d.utility.Vector3dVector(scene_point)
            scene.colors = o3d.utility.Vector3dVector(scene_color)
            vis.add_geometry(scene)

            if wcont:
                scene_cont = o3d.geometry.PointCloud()
                # scene.points = o3d.utility.Vector3dVector(scene_point)
                # scene.colors = o3d.utility.Vector3dVector(scene_color / 255.)
                vis.add_geometry(scene_cont)
                if args.w_est_cont:
                    scene_cont_est = o3d.geometry.PointCloud()
                    # scene.points = o3d.utility.Vector3dVector(scene_point)
                    # scene.colors = o3d.utility.Vector3dVector(scene_color / 255.)
                    vis.add_geometry(scene_cont_est)

            # line_set1 = []
            # sphere_list1 = []
            # line_set2 = []
            # sphere_list2 = []

            # add linesets
            line_set1 = o3d.geometry.LineSet()
            vis.add_geometry(line_set1)
            sphere_list1 = []
            jn = joints.shape[2]
            for j in range(jn):
                # spine joints
                if j % jn == 11 or j % jn == 12 or j % jn == 13:
                    continue
                if j % jn == 0:
                    r = 0.07
                else:
                    r = 0.03
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)
                vis.add_geometry(sphere)
                sphere_list1.append(sphere)

            line_set2 = o3d.geometry.LineSet()
            vis.add_geometry(line_set2)
            sphere_list2 = []
            for j in range(jn):
                # spine joints
                if j % jn == 11 or j % jn == 12 or j % jn == 13:
                    continue
                if j % jn == 0:
                    r = 0.07
                else:
                    r = 0.03
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)
                vis.add_geometry(sphere)
                sphere_list2.append(sphere)

            jlast = None
            gtlast = None
            save_imgs = False
            for i in idxs_vis:
                # add gt
                jgt = joints[bs_idx][i:i + 1]
                fn = jgt.shape[0]
                tl, jn, _ = jgt.shape
                col = his_col if i < t_his else furt_col
                add_skeleton(vis, jgt, np.array(col), line_set1, sphere_list1, jlast=gtlast)
                gtlast = np.copy(jgt)

                # plot contact
                if wcont:
                    masktmp = (is_cont[bs_idx, i]>thres).sum(axis=-1) > 0
                    pttmp = scene_point[masktmp]
                    pttmp[:, 2] = pttmp[:, 2] + 0.02
                    colortmp = np.array(cont_col * pttmp.shape[0])
                    if len(pttmp) > 0:
                        scene_cont.points = o3d.utility.Vector3dVector(pttmp)
                        scene_cont.colors = o3d.utility.Vector3dVector(colortmp)
                        vis.update_geometry(scene_cont)
                    else:
                        scene_cont.points = o3d.utility.Vector3dVector(pttmp)
                        vis.update_geometry(scene_cont)

                    if i >= t_his and args.w_est_cont:
                        masktmp = (cont_est[bs_idx, (i - t_his)]>thres).sum(axis=-1) > 0
                        pttmp = scene_point[masktmp]
                        pttmp = pttmp + 0.02
                        colortmp = np.array(cont_col_est * pttmp.shape[0])
                        if len(pttmp) > 0:
                            scene_cont_est.points = o3d.utility.Vector3dVector(pttmp)
                            scene_cont_est.colors = o3d.utility.Vector3dVector(colortmp)
                        else:
                            scene_cont_est.points = o3d.utility.Vector3dVector(pttmp)
                        vis.update_geometry(scene_cont_est)

                if i >= t_his:
                    jpred = y[bs_idx][i - t_his:i + 1 - t_his]
                    fn = jpred.shape[0]
                    tl, jn, _ = jpred.shape
                    if jlast is None:
                        add_skeleton(vis, jpred, np.array(est_col[min(i, len(est_col) - 1)]),
                                     line_set=line_set2,
                                     sphere_list=sphere_list2)
                    else:
                        add_skeleton(vis, jpred, np.array(est_col[min(i, len(est_col) - 1)]),
                                     line_set=line_set2,
                                     sphere_list=sphere_list2, jlast=jlast)

                    jlast = np.copy(jpred)

                while i == idxs_vis[0]:
                    # while True:
                    vis.poll_events()
                    vis.update_renderer()
                    key = cv2.waitKey(30)
                    # view_control.change_field_of_view(10)
                    # view_control.rotate(10,0)
                    # break
                    if key == 27: # press esc
                        save_imgs = False
                        break
                    elif key == -1:
                        continue
                    elif key == 13: # press enter
                        save_imgs = True
                        # print(key)
                        break
                if save_imgs:
                    output_path = f"{cfg.result_dir}/{'est_cont' if args.w_est_cont else 'gt_cont'}/"
                    if not os.path.exists(f"{output_path}/{item_key[bs_idx]}/"):
                        os.makedirs(f"{output_path}/{item_key[bs_idx]}/")
                    vis.capture_screen_image(f"{output_path}/{item_key[bs_idx]}/{i:02d}.jpg")
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.05)
            vis.clear_geometries()
            """save as vid"""
            if save_imgs:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                if not os.path.exists(f"{output_path}/{args.mode}/"):
                    os.makedirs(f"{output_path}/{args.mode}/")

                imgs = cv2.imread(f"{output_path}/{item_key[bs_idx]}/{i:02d}.jpg")
                out = cv2.VideoWriter(f"{output_path}/{args.mode}/{item_key[bs_idx]}.avi",
                                      fourcc, 15.0, (imgs.shape[1], imgs.shape[0]))
                for i in idxs_vis:
                    imgs = cv2.imread(f"{output_path}/{item_key[bs_idx]}/{i:02d}.jpg")
                    out.write(imgs)
                out.release()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_cont', default='gta_stage1_PVCNN2_DCT_CONT')
    parser.add_argument('--cfg', default='gta_stage2_GRU_POSE')
    parser.add_argument('--mode', default='test')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--w_est_cont', action='store_true', default=True)
    parser.add_argument('--iter', type=int, default=50)
    parser.add_argument('--iter_cont', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--cont_thre', type=float, default=0.3)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--save_joint', action='store_true', default=False)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)

    cfg = Config(f'{args.cfg}', test=args.test)
    cfg_cont = Config(f'{args.cfg_cont}', test=args.test)

    """parameter"""
    mode = args.mode
    cfg.model_specs['nk'] = nk = 1
    nz = cfg.model_specs['nz']
    rand_rot = cfg.dataset_specs['random_rot']
    t_his = cfg.dataset_specs['t_his']
    t_pred = cfg.dataset_specs['t_pred']
    t_total = t_his + t_pred
    over_all_step = 0
    cfg.dataset_specs['random_rot'] = False
    cfg.dataset_specs['step'] = args.step
    cfg.batch_size = bs = args.bs

    # get contact dct_m, idct_m
    dct_n_cont = cfg_cont.model_specs['dct_n']
    t_total = t_his + t_pred
    dct_m, idct_m = get_dct_matrix(t_total, is_torch=True)
    dct_m_cont = dct_m.to(dtype=dtype, device=device)[:dct_n_cont]
    idct_m_cont = idct_m.to(dtype=dtype, device=device)[:,:dct_n_cont]

    dct_n = cfg.model_specs['dct_n']
    dct_m = dct_m.to(dtype=dtype, device=device)[:dct_n]
    idct_m = idct_m.to(dtype=dtype, device=device)[:,:dct_n]

    """data"""
    wscene = cfg.model_specs.get('wscene', True)
    wcont = cfg.model_specs.get('wcont', True)
    cfg_cont.dataset_specs['wscene'] = True
    cfg_cont.dataset_specs['wcont'] = True
    dataset_cls = DatasetGTA if cfg.dataset == 'GTA' else None
    dataset = dataset_cls(args.mode, cfg_cont.dataset_specs)
    print(f">>> total sub sequences: {dataset.__len__()}")

    """model"""
    model = get_model(cfg)
    model.float()
    print(">>> total params: {:.5f}M".format(sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    model_cont = get_model(cfg_cont)
    model_cont.float()
    print(">>> total params in contact model: {:.5f}M".format(sum(p.numel() for p in list(model_cont.parameters())) / 1000000.0))

    if args.iter > 0:
        cp_path = cfg.model_path % args.iter
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = torch.load(cp_path, map_location='cpu')
        model.load_state_dict(model_cp['model_dict'])

    if args.iter_cont > 0:
        cp_path = cfg_cont.model_path % args.iter_cont
        print('loading contact model from checkpoint: %s' % cp_path)
        model_cp = torch.load(cp_path, map_location='cpu')
        model_cont.load_state_dict(model_cp['model_dict'])


    model.to(device)
    model.eval()
    model_cont.to(device)
    model_cont.eval()
    train(args.iter)