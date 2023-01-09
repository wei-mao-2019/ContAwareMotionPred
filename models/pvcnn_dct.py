import torch.nn.functional
from torch import nn
from pvcnn.models.prox.pvcnnpp import PVCNN2_SA, PVCNN2_FP


class PVCNN2_DCT_CONT(nn.Module):
    def __init__(self, model_specs):
        super().__init__()
        self.input_dim = input_dim = model_specs['input_dim']
        self.dct_n = dct_n = model_specs['dct_n']
        self.out_dim = out_dim = input_dim
        self.aux_dim = aux_dim = model_specs['aux_dim']
        self.nh_rnn = nh_rnn = model_specs['nh_rnn']
        self.point_feat = point_feat = model_specs['point_feat']
        self.point_extra_feat = point_extra_feat = model_specs.get('point_extra_feat',0)
        self.num_classes = num_classes = model_specs.get('num_classes',21)
        self.is_bn = is_bn = model_specs.get('is_bn',True)

        # predict the contact
        sa_blocks = [
            ((16, 2, 32), (1024, 0.15, 32, (16, 32))),
            ((32, 3, 16), (256, 0.3, 32, (32, 64))),
            ((64, 3, 8), (64, 0.6, 32, (64, 128))),
            (None, (16, 1.2, 32, (128, 128, 256))),
        ]
        fp_blocks = [
            ((128, 128), (128, 1, 8)),
            ((128, 128), (128, 1, 8)),
            ((128, 64), (64, 2, 16)),
            ((64, 64, 32), (32, 1, 32)),
        ]

        self.sa_blocks = sa_blocks = model_specs.get('sa_blocks', sa_blocks)
        self.fp_blocks = fp_blocks = model_specs.get('fp_blocks', fp_blocks)

        # encode input human poses
        self.x_enc = nn.Linear(input_dim+aux_dim, nh_rnn)
        self.x_gru = nn.GRU(nh_rnn,nh_rnn)

        # encode scene point cloud
        self.pointnet_sa = PVCNN2_SA(extra_feature_channels=num_classes*dct_n,num_classes=num_classes,
                               sa_blocks=sa_blocks, fp_blocks=fp_blocks,with_classifier=False,is_bn=is_bn)

        self.pointnet_fp = PVCNN2_FP(extra_feature_channels=num_classes*dct_n,num_classes=num_classes,
                               sa_blocks=sa_blocks, fp_blocks=fp_blocks,with_classifier=False,is_bn=is_bn,
                               sa_in_channels=self.pointnet_sa.sa_in_channels,
                               channels_sa_features=self.pointnet_sa.channels_sa_features+nh_rnn)

        self.out_mlp = nn.Sequential(nn.Linear(fp_blocks[-1][-1][0],nh_rnn),
                                        nn.Tanh(),
                                        nn.Linear(nh_rnn,num_classes*dct_n))

    def forward(self, x, scene, aux=None, cont_dct=None):
        """
        x: [seq,bs,dim]
        aux: [bs, seq, dim]
        scene: [bs, 3, N]
        cont_dct: [bs, 21*dct_n, sn]
        """
        bs, _, npts = scene.shape

        # encode x
        if aux is not None:
            hx = torch.cat([x, aux.transpose(0,1)],dim=-1)
        else:
             hx = x
        hx = self.x_enc(hx)
        hx = self.x_gru(hx)[1][0] # [bs, dim]

        hs = torch.cat([scene, cont_dct], dim=1)

        coords_list, coords, features, in_features_list = self.pointnet_sa(hs)
        features = torch.cat([features, hx[:, :, None].repeat([1, 1, features.shape[-1]])], dim=1)
        hfp = self.pointnet_fp(coords_list, coords, features, in_features_list)  # [bs,dim,npts]

        hc = self.out_mlp(hfp.transpose(1, 2)).transpose(1, 2)
        hc = hc + cont_dct
        return hc