from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import warnings
import numpy as np


warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def vali(self, vali_data, vali_loader):
        total_mse_loss = []
        total_mae_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                outputs = self.model(batch_x)
                outputs = outputs[:, -self.args.pred_len:]
                batch_y = batch_y[:, -self.args.pred_len:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                mse_loss = torch.mean((pred - true)**2)
                mae_loss = torch.mean(torch.abs(pred - true))
                total_mse_loss.append(mse_loss)
                total_mae_loss.append(mae_loss)
        total_mse_loss = np.average(total_mse_loss)
        total_mae_loss = np.average(total_mae_loss)
        self.model.train()
        return total_mse_loss, total_mae_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if self.args.loss_fn == 'MSE':
            criterion = nn.MSELoss()
        elif self.args.loss_fn == 'MAE':
            criterion = nn.L1Loss()
        opt = optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        if self.args.data_path == 'ILI.csv' or self.args.data_path == 'traffic.csv':
            sch = optim.lr_scheduler.StepLR(opt, 1, 0.7)
        else:
            sch = optim.lr_scheduler.StepLR(opt, 1, 0.85)

        for epoch in range(1, self.args.train_epochs+1):
            train_loss = []

            self.model.train()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                opt.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

                outputs = outputs[:, -self.args.pred_len:]
                batch_y = batch_y[:, -self.args.pred_len:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                opt.step()

            sch.step()
            if self.args.data_path == 'ILI.csv':
                if epoch % 12 == 0:
                    opt.param_groups[0]['lr'] = self.args.learning_rate * 0.5**(epoch//12)
            if opt.param_groups[0]['lr'] < self.args.min_lr:
                opt.param_groups[0]['lr'] = self.args.min_lr

            train_loss = np.average(train_loss)
            vali_mse_loss, vali_mae_loss = self.vali(vali_data, vali_loader)
            test_mse_loss, test_mae_loss = self.vali(test_data, test_loader)
            if self.args.loss_fn == 'MSE':
                vali_loss = vali_mse_loss
            elif self.args.loss_fn == 'MAE':
                vali_loss = vali_mae_loss

            print("Epoch: {0} | Train Loss {1:.5f} | Vali Loss {2:.5f} | Test MSE {3:.5f} MAE {4:.5f}".format(epoch, train_loss, vali_loss, test_mse_loss, test_mae_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

                outputs = outputs[:, -self.args.pred_len:]
                batch_y = batch_y[:, -self.args.pred_len:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        # f = open("result_long_term_forecast.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return
