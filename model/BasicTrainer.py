import torch
import os
import math
import copy
import time
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics
from tqdm import tqdm
from lib.data_process import get_key_from_value
from model.utils import build_anneal_beta


class Trainer(object):
    def __init__(self, model, loss, optimizer, train_dataloader, val_dataloader, test_dataloader, scaler_dict,
                 args, scheduler):
        super(Trainer, self).__init__()
        self.model = model
        self.args = args
        self.loss = loss
        self.optimizer = optimizer
        self.num_nodes_dict = args.num_nodes_dict
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.scaler_dict = scaler_dict
        self.scheduler = scheduler
        self.batch_seen = 0
        self.lam = args.lam
        self.delta = args.delta
        self.best_path = os.path.join(self.args.log_dir, self.args.save_pretrain_path)
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        if args.lamb_anneal: 
            self.lamb_list = build_anneal_beta(args.beta, args.lamb, args.epochs)
        else: 
            self.lamb_list = np.ones(args.epochs) * args.lamb

        # log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))

    def multi_train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        # train_loss_list = []
        val_loss_list = []
        t1 = time.time()
        for epoch in tqdm(range(self.args.epochs)):
            # Train
            # start_time = time.time()
            train_epoch_loss = self.multi_train_eps(epoch)
            # training_time = time.time() - start_time

            if train_epoch_loss > 1e6 or math.isnan(train_epoch_loss):
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            # Val
            val_epoch_loss = self.multi_val_epoch(epoch)
            # Best state and early stop epoch
            val_loss_list.append(val_epoch_loss)
            if val_epoch_loss < best_loss + 0.01:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False

            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.args.early_stop_patience))
                    break

            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                # self.test(self.model, self.args, self.scaler_dict, self.test_dataloader, self.logger)
                best_model = copy.deepcopy(self.model.state_dict())
                torch.save(best_model, self.best_path)
                self.logger.info("Saving current best model to " + self.best_path)
        t2 = time.time()
        # test
        self.model.load_state_dict(best_model)
        self.test(self.model, self.args, self.scaler_dict, self.test_dataloader, self.logger)
        t3 = time.time()
        self.logger.info("Total train time spent: {:.4f}".format(t2 - t1))
        self.logger.info("Total inference time spent: {:.4f}".format(t3 - t2))


    def multi_train_eps(self, epoch):
        self.model.train()
        total_loss = 0
        for inputs, targets in self.train_dataloader:
            inputs, targets = inputs.squeeze(0).to(self.args.device), targets[0].squeeze(0).to(self.args.device)
            targets_tea = targets[1].squeeze(0).to(self.args.device)
            select_dataset = get_key_from_value(self.num_nodes_dict, inputs.shape[2])
            out, (mu, std) = self.model(inputs, targets, select_dataset, batch_seen=None)
            self.optimizer.zero_grad()
            stu_loss = self.loss(out, targets[..., :self.args.output_dim], self.scaler_dict[select_dataset], alpha = self.args.alpha)
            tea_loss = self.loss(targets_tea[..., :self.args.output_dim], targets[..., :self.args.output_dim], self.scaler_dict[select_dataset])
            loss_info = -0.5*(1+2*std.log()-mu.pow(2)-std.pow(2)).sum(1).mean().div(math.log(2))

            loss_pred = (1 + self.lam) * stu_loss if (self.args.distill_loss==True and tea_loss - stu_loss < self.delta) else stu_loss
            loss = loss_pred.div(math.log(2))
            if self.args.info_loss == True:
                loss += 2 * self.lamb_list[epoch] * loss_info
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm).item()
            self.optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(self.train_dataloader)
        current_lr = self.optimizer.param_groups[0]['lr']
        if self.args.lr_decay:
            self.scheduler.step()
        self.logger.info(
            "  train loss is:  " + str(total_loss) +"  current_lr is:  " + str(
                current_lr) + " grad_norm is:  " + str(grad_norm)
            )
        return train_loss

    def multi_val_epoch(self, epoch):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            mae = 0
            rmse = 0
            mape = 0
            total_count = 0
            total_mape_count = 0
            total_batch = 0
            for inputs, targets in self.val_dataloader:
                inputs, targets = inputs.squeeze(0).to(self.args.device), targets[0].squeeze(0).to(self.args.device)
                select_dataset = get_key_from_value(self.num_nodes_dict, inputs.shape[2])
                out, (mu, std) = self.model(inputs, targets, select_dataset, batch_seen=None)
                y_lbl = targets[..., :self.args.output_dim]
                stu_loss = self.loss(out, y_lbl, self.scaler_dict[select_dataset])
                loss_pred = stu_loss
                loss = loss_pred.div(math.log(2))
                # if self.args.info_loss == True:
                #     loss += 2 * self.lamb_list[epoch] * loss_info
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                val_loss = total_val_loss / len(self.val_dataloader)
                if self.args.real_value == False:
                    out = self.scaler_dict[select_dataset].inverse_transform(out)
                    y_lbl = self.scaler_dict[select_dataset].inverse_transform(targets[..., :self.args.output_dim])
                else:
                    y_lbl = targets[..., :self.args.output_dim]


                batch_mae, batch_rmse, batch_mape, batch_mse, corr, mae_count, rmse_count, mse_count, mape_count = \
                    All_Metrics(out, y_lbl, self.args.mae_thresh, self.args.thresh)
                mae += batch_mae * mae_count
                rmse += batch_mse * rmse_count
                mape += batch_mape * mape_count
                total_count += mae_count
                total_mape_count += mape_count
                total_batch += len(y_lbl)
        mae /= total_count
        rmse = (rmse / total_count) ** 0.5
        mape /= total_mape_count


                
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        self.logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%, CORR:{:.4f}".format(
            mae, rmse, mape * 100, corr))
        return val_loss

    @staticmethod
    def test(model, args, scaler_dict, test_dataloader, logger, path=None):
        if path != None:
            if torch.cuda.device_count() > 1:
                model.load_state_dict(torch.load(path))
            else:
                model_weights = {k.replace('module.', ''): v for k, v in torch.load(path).items()}
                model.load_state_dict(model_weights)
            model.to(args.device)
        model.eval()
        trues = []
        preds = []
        mae = 0
        rmse = 0
        mape = 0
        total_count = 0
        total_mape_count = 0
        total_batch = 0
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.squeeze(0).to(args.device), targets[0].squeeze(0).to(args.device)
                select_dataset = get_key_from_value(args.num_nodes_dict, inputs.shape[2])
                output,(mu,std) = model(inputs, targets, select_dataset, batch_seen=None)
                if args.real_value == False:
                    output = scaler_dict[select_dataset].inverse_transform(output)
                    y_lbl = scaler_dict[select_dataset].inverse_transform(targets[..., :args.output_dim])
                else:
                    y_lbl = targets[..., :args.output_dim]
                trues.append(y_lbl)
                preds.append(output)
                batch_mae, batch_rmse, batch_mape, batch_mse, corr, mae_count, rmse_count, mse_count, mape_count = \
                    All_Metrics(output, y_lbl, args.mae_thresh, args.beta)
                mae += batch_mae * mae_count
                rmse += batch_mse * rmse_count
                mape += batch_mape * mape_count
                total_count += mae_count
                total_mape_count += mape_count
                total_batch += len(y_lbl)
                if args.model == 'OpenCity':
                    print(total_batch, batch_mae, batch_rmse, batch_mape, total_count, total_mape_count)
        mae /= total_count
        rmse = (rmse / total_count) ** 0.5
        mape /= total_mape_count
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%, CORR:{:.4f}".format(
            mae, rmse, mape * 100, corr))
        trues, preds = torch.cat(trues, dim=0), torch.cat(preds, dim=0)
        mae, rmse = [], []
        for t in range(trues.shape[1]):
            batch_mae, batch_rmse, batch_mape, batch_mse, corr, mae_count, rmse_count, mse_count, mape_count = \
                All_Metrics(preds[:, t, ...], trues[:, t, ...], args.mae_thresh, args.beta)
            mae.append(round(batch_mae.item(),2))
            rmse.append(round(batch_rmse.item(),2))
            log = "Horizon {:02d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%, Corr: {:.4f}".format(
                t + 1, batch_mae, batch_rmse, batch_mape * 100, corr)
            logger.info(log)
        logger.info(mae)
        logger.info(rmse)
        batch_mae, batch_rmse, batch_mape, batch_mse, corr, mae_count, rmse_count, mse_count, mape_count = \
            All_Metrics(preds, trues, args.mae_thresh, args.beta)

        mae, rmse = [], []
        if args.dataset_use == ['NYC_TAXI']:
            region_list = [41,142,162,235]
        else:
            region_list = [55,27,75,7]
        for s in region_list:
            batch_mae, batch_rmse, batch_mape, batch_mse, corr, mae_count, rmse_count, mse_count, mape_count = \
                All_Metrics(preds[:, :,s, ...], trues[:, :,s, ...], args.mae_thresh, args.beta)
            mae.append(round(batch_mae.item(),2))
            rmse.append(round(batch_rmse.item(),2))
            log = "Region {:02d}, Mean: {:.1f}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%, Corr: {:.4f}".format(
                s + 1, trues[:, :,s, ...].mean().item(), batch_mae, batch_rmse, batch_mape * 100, corr)
            logger.info(log)
        logger.info(mae)
        logger.info(rmse)
