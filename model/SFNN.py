import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.need_norm  = configs.need_norm
        self.period = configs.period
        self.dropout = nn.Dropout(configs.dropout)
        self.inp_proj = nn.Linear(configs.seq_len, configs.seq_len)
        self.layers1 = nn.ModuleList([nn.Linear(configs.seq_len, configs.seq_len) for _ in range(configs.n_layers)])
        if configs.mixer:
            self.layers2 = nn.ModuleList([nn.Linear(configs.n_series, configs.n_series) for _ in range(configs.n_layers)])
        else:
            self.layers2 = None
        if self.need_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm([configs.n_series, configs.seq_len]) for _ in range(configs.n_layers)])
        self.out_proj = nn.Linear(configs.seq_len, configs.pred_len)

    def forward(self, x):
        if self.need_norm:
            mean = x[:, -self.period:].mean(dim=1, keepdim=True)
            x = x - mean
        x = x.transpose(1, -1)
        x = self.inp_proj(x)
        for i, layer in enumerate(self.layers1):
            x = F.relu(layer(self.dropout(x))) + x
            if self.layers2 is not None:
                x = F.selu(self.layers2[i](self.dropout(x))) + x
            if self.need_norm:
                x = self.layer_norms[i](x)
        x = self.out_proj(x)
        x = x.transpose(1, -1)
        if self.need_norm:
            x = x + mean
        return x
