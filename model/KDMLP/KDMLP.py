from numbers import Number
import torch.nn as nn
import torch.nn.functional as F
import torch
from KDMLP.mlp import MultiLayerPerceptron


class KDMLP(nn.Module):
    def __init__(
        self,
        args_predictor,
    ):

        super(KDMLP, self).__init__()
        self.num_layers = args_predictor.num_layers
        self.input_window = args_predictor.input_window
        self.output_window = args_predictor.output_window
        self.input_dim = args_predictor.input_dim
        self.node_dim = args_predictor.node_dim
        self.embed_dim = args_predictor.embed_dim
        self.temp_dim_tid = args_predictor.temp_dim_tid
        self.temp_dim_diw = args_predictor.temp_dim_diw
        self.num_nodes = args_predictor.num_nodes
        hidden_dims = []
        hidden_dims.append(self.embed_dim+self.temp_dim_tid + self.temp_dim_diw)

        hidden_dims.append(self.node_dim)
        # hidden_dims.append(self.node_dim)
        self.hidden_dim = sum(hidden_dims)
        self.K = int(self.hidden_dim // 2)
        self.num_sample = args_predictor.num_sample
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)])
                # embedding layer

        self.time_in_day_emb = nn.Parameter(
                torch.empty(48, self.temp_dim_tid))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(
                torch.empty(7, self.temp_dim_diw))
        nn.init.xavier_uniform_(self.day_in_week_emb)
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_window, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        self.regression_layer = nn.Conv2d(
            in_channels=self.K, out_channels=self.output_window, kernel_size=(1, 1), bias=True)
        
        self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
        nn.init.xavier_uniform_(self.node_emb)

        # self.nodevec_p1 = nn.Parameter(torch.randn(48, self.node_dim).to(args.device), requires_grad=True).to(args.device)
        # self.nodevec_p2 = nn.Parameter(torch.randn(self..num_nodes, self.node_dim).to(args.device), requires_grad=True).to(args.device)
        # self.nodevec_pk = nn.Parameter(torch.randn(self.node_dim, self.node_dim, self.node_dim).to(args.device), requires_grad=True).to(args.device)
        # self.dne_emb_layer = nn.Conv2d(
        # in_channels= self.input_window, out_channels=1, kernel_size=(1, 1), bias=True)
        
        # self.dne_act = nn.Softmax(dim=2)
    
    def reparametrize_n(self, mu, std, n=1):
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1).to(mu.device)  # Ensure expanded tensor is on the same device as mu
            else:
                return v.expand(n, *v.size())

        if n != 1:
            mu = expand(mu)
            std = expand(std)

        # Generate eps on the same device as std (or mu)
        eps = torch.normal(0, 1, size=std.size(), device=std.device)  # Directly set the device here
        return mu + eps * std

    def forward(self, feats, select_dataset):
        X = feats[..., range(3)]
        
        t_i_d_data   = feats[..., -2] # B, L, N
        d_i_w_data   = feats[..., -1] # B, L, N

        T_i_D_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :]).type(torch.LongTensor)]    # [B, N, D]
        D_i_W_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]    # [B, N, D]

        B, L, N, _ = X.shape # B, L, N, 1
        X = X.transpose(1, 2).contiguous()                      # B, N, L, 1
        h = X.view(B, N, -1).transpose(1, 2).unsqueeze(-1)      # B, L*3, N, 1
        time_series_emb = self.time_series_emb_layer(h)         # B, D, N, 1

        node_emb = []
        node_emb.append(self.node_emb.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1))  # B, D, N, 1
        # dne = self.construct_dne(kwargs.get('te').type(torch.LongTensor))
        # node_emb.append(dne)
        
        # temporal embeddings
        tem_emb  = []
        if T_i_D_emb is not None:
            tem_emb.append(T_i_D_emb.transpose(1, 2).unsqueeze(-1))                     # B, D, N, 1
        if D_i_W_emb is not None:
            tem_emb.append(D_i_W_emb.transpose(1, 2).unsqueeze(-1))                     # B, D, N, 1
        
        # concate all embeddings
        h = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)

        h = self.encoder(h) # B, 256, N, 1
        mu = h[:,:self.K, :,:]
        std = F.softplus(h[:,self.K:,:,:], beta=1)
        h = self.reparametrize_n(mu, std, self.num_sample)
        if self.num_sample == 1 : pass
        elif self.num_sample > 1 : h = h.mean(0)
        pred = self.regression_layer(h)
        return pred, (mu.permute(0, 1, 3, 2), std.permute(0, 1, 3, 2))




