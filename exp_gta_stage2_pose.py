import math
import argparse
import time
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchgeometry
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

from utils.config import Config
from datasets.dataset_gta import DatasetGTA
from models.motion_pred import *
from utils import *
from utils.util import *

def loss_function(joints, y, root_traj, root_joint_idx,
                  cont_point, epoch, cfg, **kwargs):
    """
    joints: [bs,nk,t_total,jn,3]
    y: [bs,nk,t_pred,jn,dim]
    scene_vert: [bs, nk, npts, 3]
    rand_rot: [bs, nk, 4, 4]
    """

    bs, _, jn, _ = y.shape
    t_his = cfg.dataset_specs['t_his']
    t_pred = cfg.dataset_specs['t_pred']
    wscene = cfg.dataset_specs['wscene']
    wcont = cfg.dataset_specs['wcont']
    ws = cfg.model_specs['lambdas']
    se = cfg.model_specs.get('start_epochs', [0]*len(ws))
    jidx = np.setdiff1d(np.arange(jn),[root_joint_idx])

    #[bs, nk, fn, 21, 3]
    loss_joints = torch.mean((joints[:, t_his:][:,:,jidx] - y[:,:,jidx]).pow(2).sum(dim=-1))
    loss_root = (joints[:,:,root_joint_idx]-root_traj).pow(2).sum(dim=-1).mean()
    loss_all = (ws[0] if epoch >= se[0] else 0) * (loss_joints + loss_root)

    yorig = root_traj[:,t_his:,None].to(dtype=y.dtype)
    yt = y.clone()
    yt[:,:,root_joint_idx:root_joint_idx+1] = yorig
    yt[:, :, jidx] = yt[:, :, jidx] + yorig

    if wcont:
        loss_cont = torch.mean((yt - cont_point[...,:3]).pow(2).sum(dim=-1) * cont_point[...,3])

        loss_all = loss_all + (ws[1] if epoch >= se[1] else 0) * loss_cont
    else:
        loss_cont = torch.tensor(0,dtype=loss_all.dtype, device=loss_all.device)

    return loss_all, np.array([loss_all.item(), loss_joints.item(), loss_root.item(), loss_cont.item()])


def train(epoch, gpu, model,optimizer,scheduler,dataset, cfg, args,
          tb_logger=None,logger=None,
          device=None,dtype=None,
          data_sampler=None, scaler=None,
          **kwargs):
    is_rand_rot = cfg.dataset_specs['random_rot']
    t_his = cfg.dataset_specs['t_his']
    t_pred = cfg.dataset_specs['t_pred']
    wscene = cfg.dataset_specs['wscene']
    wcont = cfg.dataset_specs['wcont']
    root_joint_idx = 14
    root_idx = [root_joint_idx*3,root_joint_idx*3+1,root_joint_idx*3+2]

    t_s = time.time()
    train_losses = 0
    train_grad = 0
    total_num_sample = 0

    loss_names = ['ALL', 'joint', 'root_joint', 'cont']

    if args.is_dist:
        generator = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                               num_workers=0, pin_memory=True, sampler=data_sampler)
    else:
        generator = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                           num_workers=2, pin_memory=True, drop_last=True)

    for pose, scene_vert, scene_origin, _, item_key in (tqdm(generator) if gpu==0 or not args.is_dist else generator):
        with torch.no_grad():
            bs = pose.shape[0]
            nj = pose.shape[2]
            joints = pose.to(device=device)
            if wscene or wcont:
                scene_vert = scene_vert.to(device=device)#[:,:10000]

            if is_rand_rot:
                ang = torch.zeros([bs, 3],device=device,dtype=dtype)
                ang[:,2] = torch.rand(bs,device=device,dtype=dtype)*math.pi
                rand_rot = torchgeometry.angle_axis_to_rotation_matrix(ang).unsqueeze(1)
                rand_tran = torch.rand([bs,3],device=device,dtype=dtype)-0.5
                rand_rot[:,0,:3,3] = rand_tran
                if wscene or wcont:
                    scene_vert = torch.matmul(rand_rot[:,0,:3,:3],scene_vert.transpose(1,2)).transpose(1,2) \
                                 + rand_rot[:,:1,:3,3]
                joints = torch.matmul(rand_rot[:,:,:3,:3],joints.transpose(2,3)).transpose(2,3) \
                             + rand_rot[:,:1,None,:3,3]
            else:
                rand_rot = None

            if wcont:
                dist = (scene_vert[:,None,:,None,:]-joints[:,:,None,:,:]).norm(dim=-1)
                min_dist_value = (dist.min(dim=2)[0]<0.3).to(dtype=dtype)
                min_dist_idx = dist.min(dim=2)[1].reshape([-1])
                idx_tmp = torch.arange(bs,device=device)[:,None].repeat([1,(t_pred+t_his)*nj]).reshape([-1])

                cont_points = scene_vert[idx_tmp,min_dist_idx,:].reshape([bs,t_his+t_pred,nj,3])
                cont_points = cont_points * min_dist_value[...,None]
                cont_points = torch.cat([cont_points,min_dist_value[...,None]],dim=-1)

            joints_orig = joints[:, :, 14:15]
            joints = joints - joints_orig
            joints[:, :, 14:15] = joints_orig

        with autocast(enabled=args.is_amp):

            # def forward(self, x, cont, cont_mask, aux=None, horizon=30,
            #             dct_m=None, idct_m=None, root_idx=None):
            y, root_traj = model(joints[:,:t_his].reshape([bs,t_his,-1]).transpose(0,1),
                                 cont_points[:,t_his:].reshape([bs,t_pred,-1]) if wcont else None,
                                 None, None, t_pred, dct_m=kwargs['dct_m'],idct_m=kwargs['idct_m'],
                                 root_idx=root_idx)

            y = y.transpose(0,1)

            y = y.reshape([bs, t_pred,-1,3])

            # def loss_function(joints, y, root_traj, root_joint_idx,
            #                   cont_point, epoch, cfg, **kwargs):
            loss, losses = loss_function(joints, y, root_traj, root_joint_idx,
                                         cont_points[:,t_his:] if wcont else None, epoch, cfg,
                                         **kwargs)

        optimizer.zero_grad()

        if args.is_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=1000)
        grad_norm = 0
        train_grad += grad_norm
        train_losses += losses
        total_num_sample += 1
        # break
    scheduler.step()
    train_losses /= total_num_sample
    lr = optimizer.param_groups[0]['lr']
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])

    # average cost of log time 20s
    if gpu == 0 or not args.is_dist:
        tb_logger.add_scalar('train_grad', train_grad / total_num_sample, epoch)
        for name, loss in zip(loss_names, train_losses):
            tb_logger.add_scalars('' + name, {'train': loss}, epoch)

        logger.info('====> Epoch: {} Time: {:.2f} {} lr: {:.5f}'.format(epoch, time.time() - t_s, losses_str, lr))

def main(gpu, cfg, args):
    if args.is_dist:
        rank = args.nr * args.gpus + gpu
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)


    params = {}

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    torch.cuda.set_device(gpu)
    device = torch.device('cuda', index=gpu)
    params['device'] = device
    params['dtype'] = dtype

    if not args.is_dist:
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu)

    if args.is_amp:
        scaler = GradScaler()
        params['scaler'] = scaler

    if gpu == 0 or not args.is_dist:
        tb_logger = SummaryWriter(cfg.tb_dir) if args.mode == 'train' else None
        logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
        params['tb_logger'] = tb_logger
        params['logger'] = logger

    """parameter"""
    mode = args.mode

    t_his = cfg.dataset_specs['t_his']
    t_pred = cfg.dataset_specs['t_pred']
    dct_n = cfg.model_specs['dct_n']
    t_total = t_his + t_pred
    dct_m, idct_m = get_dct_matrix(t_total, is_torch=True)
    dct_m = dct_m.to(dtype=dtype, device=device)[:dct_n]
    idct_m = idct_m.to(dtype=dtype, device=device)[:,:dct_n]
    params['dct_m'] = dct_m
    params['idct_m'] = idct_m

    """data"""
    cfg.dataset_specs['wscene'] = cfg.model_specs.get('wscene',True)
    cfg.dataset_specs['wcont'] = cfg.model_specs.get('wcont',True)
    cfg.dataset_specs['cont_type'] = cfg.model_specs.get('cont_type', 'perframe_contact')
    dataset_cls = DatasetGTA if cfg.dataset == 'GTA' else None
    dataset = dataset_cls(args.mode, cfg.dataset_specs)

    if args.is_dist:
        data_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                       num_replicas=args.world_size,
                                                                       rank=rank)
        params['data_sampler'] = data_sampler
    if gpu == 0 or not args.is_dist:
        logger.info(f">>> total sub sequences: {dataset.__len__()}")

    """model"""
    model = get_model(cfg).to(dtype=dtype, device=device)
    if args.is_dist:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-3)
    scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.num_epoch_fix, nepoch=cfg.num_epoch)

    if args.iter > 0:
        cp_path = cfg.model_path % args.iter
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = torch.load(cp_path, map_location=device)
        optimizer.load_state_dict(model_cp['opt_dict'])
        scheduler.load_state_dict(model_cp['scheduler_dict'])
        model.load_state_dict(model_cp['model_dict'])
    if args.is_dist:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    if gpu == 0 or not args.is_dist:
        logger.info(">>> total params: {:.5f}M".format(sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    if mode == 'train':
        model.train()
        for i in range(args.iter, cfg.num_epoch):
            train(i, gpu, model,optimizer,scheduler,dataset, cfg, args, **params)

            if cfg.save_model_interval > 0 \
                    and (i + 1) % cfg.save_model_interval == 0 \
                    and (gpu == 0 or not args.is_dist ):
                cp_path = cfg.model_path % (i + 1)
                model_cp = {'model_dict': model.module.state_dict() if args.is_dist else model.state_dict(),
                            'opt_dict': optimizer.state_dict(),
                            'scheduler_dict': scheduler.state_dict()}
                torch.save(model_cp,cp_path)
            if args.is_dist:
                dist.barrier()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='gta_stage2_GRU_POSE')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--is_amp', action='store_true', default=True)

    parser.add_argument('--is_dist', action='store_true', default=False)
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=4, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')

    args = parser.parse_args()
    cfg = Config(f'{args.cfg}', test=args.test)

    if args.is_dist:
        args.world_size = args.gpus * args.nodes
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '8888'
        mp.spawn(main, nprocs=args.gpus, args=(cfg, args))
    else:
        main(args.gpu_index, cfg, args)
