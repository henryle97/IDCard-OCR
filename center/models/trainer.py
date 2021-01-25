from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import torch
from progress.bar import Bar
from torch.nn.parallel import DataParallel
from .utils import _sigmoid
from utils.utils import AverageMeter
from .losses import FocalLoss, RegL1Loss, RegLoss
from .model import load_model, create_model, save_model
from dataset.dataset_custom import DATASET_CUSTOM
from utils.logger import Logger


class Trainer(object):
    def __init__(self, config):

        self.arch = config['model']['arch']
        self.heads = config['model']['heads']
        self.head_conv = config['model']['head_conv']
        self.lr = config['train']['lr']
        self.lr_step = config['train']['lr_step']
        self.resume_path = config['train']['resume_path']
        self.seed = config['train']['seed']
        self.not_cuda_benchmark = config['train']['not_cuda_benchmark']

        self.gpu_str = config['gpu_str']
        self.gpus = config['gpus']
        self.chunk_sizes = config['train']['chunk_sizes']

        self.num_epochs = config['train']['num_epochs']
        self.batch_size = config['train']['batch_size']
        self.num_workers = config['train']['num_workers']
        self.val_intervals = config['train']['val_intervals']
        self.save_all = config['train']['save_all']
        self.metric = config['train']['metric']
        self.save_dir = config['save_dir']
        self.num_iters = config['train']['num_iters']
        self.exp_id = config['exp_id']
        self.print_iter = config['train']['print_iter']
        self.hide_data_time = config['train']['hide_data_time']

        self.model = create_model(self.arch, self.heads, self.head_conv)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.loss_stats, self.loss = self._get_losses(config)
        self.model_with_loss = ModelWithLoss(self.model, self.loss)

        if len(self.gpus) == 1 and self.gpu_str != '-1':
            torch.cuda.set_device(int(self.gpus[0]))
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.gpu_str != '-1' else 'cpu')

        self.set_device(self.gpus, self.chunk_sizes, self.device)

        self.start_epoch = 0
        if self.resume_path != '':
            self.model, self.optimizer, self.start_epoch = load_model(
                self.model, self.resume_path, self.optimizer, self.lr, self.lr_step)


        self.train_loader = torch.utils.data.DataLoader(
            DATASET_CUSTOM(config, split='train'),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        self.val_loader = torch.utils.data.DataLoader(
            DATASET_CUSTOM(config, split='val'),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        self.logger = Logger(config)

    def train(self):
        print('Starting training...')
        best = 1e10
        for epoch in range(self.start_epoch + 1, self.num_epochs + 1):
            log_dict_train, _ = self.run_epoch('train', epoch, self.train_loader)
            # Logger
            mark = epoch if self.save_all else 'last'
            self.logger.write('epoch: {} |'.format(epoch))
            for k, v in log_dict_train.items():
                self.logger.scalar_summary('train_{}'.format(k), v, epoch)
                self.logger.write('{} {:8f} | '.format(k, v))

            # Validation
            if self.val_intervals > 0 and epoch % self.val_intervals == 0:
                save_model(os.path.join(self.save_dir, 'model_{}.pth'.format(mark)),
                           epoch, self.model, self.optimizer)

                with torch.no_grad():
                    log_dict_val, preds = self.run_epoch('val', epoch, self.val_loader)
                for k, v in log_dict_val.items():
                    self.logger.scalar_summary('val_{}'.format(k), v, epoch)
                    self.logger.write('{} {:8f} | '.format(k, v))
                if log_dict_val[self.metric] < best:  # best loss
                    best = log_dict_val[self.metric]
                    save_model(os.path.join(self.save_dir, 'model_best.pth'),
                               epoch, self.model)
            else:
                save_model(os.path.join(self.save_dir, 'model_last.pth'),
                           epoch, self.model, self.optimizer)
            self.logger.write('\n')

            # Scale learning rate
            if epoch in self.lr_step:
                save_model(os.path.join(self.save_dir, 'model_{}.pth'.format(epoch)),
                           epoch, self.model, self.optimizer)
                self.lr = self.lr * (0.1 ** (self.lr_step.index(epoch) + 1))
                print('Drop LR to', self.lr)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
        self.logger.close()


    def set_device(self, gpus, chunk_sizes, device):

        print("Set device: ", gpus, device)
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus).to(device)
        else:
            # print("Push model to cpu or one gpu ")
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)


    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss

        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if self.num_iters < 0 else self.num_iters
        bar = Bar('{}'.format(self.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    if k == 'input':
                        batch[k] = batch[k].to(device=self.device, non_blocking=True, dtype=torch.float32)
                    else:
                        batch[k] = batch[k].to(device=self.device, non_blocking=True)
            # print(batch)
            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)

            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

            if not self.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if self.print_iter > 0:
                if iter_id % self.print_iter == 0:
                    print('{}| {}'.format(self.exp_id, Bar.suffix))
            else:
                bar.next()

            # if opt.debug > 0:
            #     self.debug(batch, output, iter_id)
            #
            # if opt.test:
            #     self.save_result(output, batch, results)
            del output, loss, loss_stats

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def _get_losses(self, config):
        loss_states = ['loss', 'hm_loss', 'off_loss']
        loss = CombineLoss(config)
        return loss_states, loss


class CombineLoss(torch.nn.Module):
  def __init__(self, config):
    super(CombineLoss, self).__init__()
    self.crit = FocalLoss()
    self.crit_reg = RegL1Loss() if config['train']['reg_loss'] == 'l1' else \
              RegLoss() if config['train']['reg_loss'] == 'sl1' else None

    self.config = config

  def forward(self, outputs, batch):
    config = self.config
    hm_loss, off_loss = 0, 0
    num_stacks = config['train']['num_stacks']
    for s in range(num_stacks):
      output = outputs[s]
      output['hm'] = _sigmoid(output['hm'])
      hm_loss += self.crit(output['hm'], batch['hm']) / num_stacks
      off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                             batch['ind'], batch['reg']) / num_stacks

    loss = config['train']['hm_weight'] * hm_loss + config['train']['off_weight'] * off_loss
    loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'off_loss': off_loss}

    return loss, loss_stats


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model(batch['input'])
        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats


