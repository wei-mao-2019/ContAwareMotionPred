import torch.nn.functional
from torch import nn

class GRU_POSE(nn.Module):
    def __init__(self, model_specs):
        super().__init__()
        self.input_dim = input_dim = model_specs['input_dim']
        self.cont_dim = cont_dim = model_specs.get('cont_dim', 84)
        self.out_dim = out_dim = input_dim
        self.aux_dim = aux_dim = model_specs.get('aux_dim', 0)
        self.nh_rnn = nh_rnn = model_specs['nh_rnn']
        self.dct_n = dct_n = model_specs['dct_n']
        self.root_net_is_bn = root_net_is_bn = model_specs.get('root_net_is_bn',False)
        self.root_net_resblock = root_net_resblock = model_specs.get('root_net_resblock',2)
        self.point_feat = point_feat = model_specs['point_feat']
        self.point_extra_feat = point_extra_feat = model_specs.get('point_extra_feat',0)
        self.wscene = wscene = model_specs.get('wscene',True)
        self.wcont = wcont = model_specs.get('wcont',True)

        # encode input human poses
        self.x_enc = nn.Linear(input_dim+aux_dim, nh_rnn)

        # encode pose sequences
        self.x_gru = nn.GRU(nh_rnn,nh_rnn)

        # encode contact
        if wcont:
            self.cont_enc = nn.Linear(cont_dim, nh_rnn)
            self.cont_gru = nn.GRU(nh_rnn,nh_rnn)

        # predict global trajectory via dct
        if not root_net_is_bn:
            if wcont:
                self.g_w1 = nn.Sequential(nn.Linear(dct_n*3+nh_rnn+nh_rnn,nh_rnn),
                                          nn.Tanh())
            else:
                self.g_w1 = nn.Sequential(nn.Linear(dct_n*3+nh_rnn,nh_rnn),
                                          nn.Tanh())
            self.g_w2 = nn.Sequential(nn.Linear(nh_rnn,nh_rnn),
                                      nn.Tanh(),
                                      nn.Linear(nh_rnn,nh_rnn),
                                      nn.Tanh())
            self.g_w3 = nn.Sequential(nn.Linear(nh_rnn,nh_rnn),
                                      nn.Tanh(),
                                      nn.Linear(nh_rnn,nh_rnn),
                                      nn.Tanh())
            self.g_w4 = nn.Linear(nh_rnn,dct_n*3)
        else:
            if wcont:
                self.g_w1 = nn.Sequential(nn.Linear(dct_n*3+nh_rnn+nh_rnn,nh_rnn),
                                          nn.BatchNorm1d(nh_rnn),
                                          nn.Tanh())
            else:
                self.g_w1 = nn.Sequential(nn.Linear(dct_n*3+nh_rnn,nh_rnn),
                                          nn.BatchNorm1d(nh_rnn),
                                          nn.Tanh())
            self.g_w2 = nn.Sequential(nn.Linear(nh_rnn,nh_rnn),
                                      nn.BatchNorm1d(nh_rnn),
                                      nn.Tanh(),
                                      nn.Linear(nh_rnn,nh_rnn),
                                      nn.BatchNorm1d(nh_rnn),
                                      nn.Tanh())
            self.g_w3 = nn.Sequential(nn.Linear(nh_rnn,nh_rnn),
                                      nn.BatchNorm1d(nh_rnn),
                                      nn.Tanh(),
                                      nn.Linear(nh_rnn,nh_rnn),
                                      nn.BatchNorm1d(nh_rnn),
                                      nn.Tanh())
            self.g_w4 = nn.Linear(nh_rnn,dct_n*3)


        # encode temporal information
        out_in_dim = aux_dim + out_dim

        if wcont:
            out_in_dim += cont_dim

        self.out_enc = nn.Linear(out_in_dim+3, nh_rnn)

        # decode pose sequence
        self.out_gru = nn.GRUCell(nh_rnn, nh_rnn)

        # decode pose
        self.out_mlp = nn.Linear(nh_rnn, out_dim)

    def forward(self, x, cont, cont_mask, aux=None, horizon=30,
                dct_m=None, idct_m=None, root_idx=None):
        """
        x: [seq,bs,dim]
        scene: [bs, 3, N]
        cont: bs, seq, nj*4
        aux: [bs,seq,...] or None
        """
        t_his, bs, nfeat = x.shape

        # encode x
        if aux is not None:
            hx = torch.cat([x,aux[:,:x.shape[0]].transpose(0,1)],dim=-1)
        else:
            hx = x
        hx = self.x_enc(hx)
        hx = self.x_gru(hx)[1][0]

        # encode cont
        if self.wcont:
            cont_tmp = cont.clone()
            # cont_tmp[:,:,:56] = 0
            hcont = self.cont_enc(cont_tmp)
            hcont = self.cont_gru(hcont.transpose(0,1))[1][0]

        # predict global traj
        pad_idx = list(range(t_his))+[t_his-1]*horizon
        root_his = x[:,:,root_idx][pad_idx].transpose(0,1)
        root_his_dct = torch.matmul(dct_m[None],root_his).reshape([bs,-1])

        if self.wcont:
            root_h = self.g_w1(torch.cat([root_his_dct,hx,hcont],dim=1))
        else:
            root_h = self.g_w1(torch.cat([root_his_dct,hx],dim=1))

        root_h = self.g_w2(root_h) + root_h
        root_h = self.g_w3(root_h) + root_h
        root_pred = self.g_w4(root_h) + root_his_dct
        root_traj = torch.matmul(idct_m[None],root_pred.reshape([bs,-1,len(root_idx)])) #[bs,t_total,3]

        y = []
        ylast = x[-1]
        for i in range(horizon):
            ylast = torch.cat([ylast, root_traj[:,t_his+i].to(dtype=ylast.dtype)], dim=-1)
            if aux is not None:
                ylast = torch.cat([ylast, aux[:,t_his+i].to(dtype=ylast.dtype)], dim=1)

            if self.wcont:
                ylast = torch.cat([ylast, cont[:, i]], dim=1)

            hy = self.out_enc(ylast)
            hy = self.out_gru(hy, hx)
            ylast = ylast[:,:self.out_dim] + self.out_mlp(hy)
            y.append(ylast[None,:,:])

        y = torch.cat(y,dim=0)
        return y, root_traj