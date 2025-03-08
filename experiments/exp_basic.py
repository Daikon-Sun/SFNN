import os
import torch
from model import SFNN
import numpy as np
from scipy import stats


def ma(x, window):
    cumsum = np.cumsum(x, axis=0)
    result = np.empty((len(x) - window + 1, x.shape[1]))
    result[0] = cumsum[window-1] / window
    result[1:] = (cumsum[window:] - cumsum[:-window]) / window
    return result

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {'SFNN': SFNN}
        self.device = self._acquire_device()

        train_data, train_loader = self._get_data(flag='train')
        ma_train_data = ma(train_data.data_x, 2*self.args.period)
        prv_avg = ma_train_data[:-2*self.args.period].flatten()
        nxt_avg = ma_train_data[2*self.args.period:].flatten()
        pearson = stats.pearsonr(nxt_avg, prv_avg).statistic
        self.args.need_norm = (pearson >= 0.7)
        print(f'pearson: {pearson:.4f}')
        self.args.n_series = train_data.data_x.shape[1]

        self.model = self._build_model().to(self.device)
        n_para = sum(p.numel() for p in self.model.parameters())
        print(f'number of parameters: {n_para}')

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        return torch.device('cuda')

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
