import yaml
import os
import shutil


class Config:

    def __init__(self, cfg_id, test=False):
        self.id = cfg_id
        cfg_name = 'cfg/%s.yml' % cfg_id
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)
        cfg = yaml.safe_load(open(cfg_name, 'r'))

        # create dirs
        self.base_dir = '/tmp' if test else 'results'

        self.cfg_dir = '%s/%s' % (self.base_dir, cfg_id)
        self.model_dir = '%s/models' % self.cfg_dir
        self.result_dir = '%s/results' % self.cfg_dir
        self.log_dir = '%s/log' % self.cfg_dir
        self.tb_dir = '%s/tb' % self.cfg_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)

        shutil.copyfile(cfg_name, '%s/%s.yml' % (self.cfg_dir,cfg_id))

        # common
        self.dataset = cfg.get('dataset', 'h36m')
        self.batch_size = cfg.get('batch_size', 8)
        self.save_model_interval = cfg.get('save_model_interval', 1)
        self.dataset_specs = cfg.get('dataset_specs', dict())

        self.lr = cfg['lr']
        self.num_epoch = cfg['num_epoch']
        self.num_epoch_fix = cfg.get('num_epoch_fix', self.num_epoch)
        self.model_path = os.path.join(self.model_dir, '%04d.p')
        self.model_name = cfg.get('model_name', '')
        self.model_specs = cfg.get('model_specs', dict())
        self.dataset_specs = cfg.get('dataset_specs', dict())
