import os
import random
import logging
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from tqdm import tqdm
from math import ceil
import numpy as np
from distutils.version import LooseVersion
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import warnings
import sys

sys.path.append(os.path.abspath('.'))
from utils.eval import Eval
from utils.train_helper import get_model

from datasets.cityscapes_Dataset import City_Dataset, City_DataLoader, inv_preprocess, decode_labels
from datasets.gta5_Dataset import GTA5_DataLoader, GTA5_xuanran_DataLoader, Mix_DataLoader
from datasets.synthia_Dataset import SYNTHIA_DataLoader

datasets_path = {
    'cityscapes': {'data_root_path': '../../DATASETS/datasets_original/Cityscapes', 'list_path': '../datasets/city_list',
                   'image_path': '../../DATASETS/datasets_original/Cityscapes/leftImg8bit',
                   'gt_path': '../../DATASETS/datasets_original/Cityscapes/gtFine'},
    'gta5': {'data_root_path': '../../DATASETS/datasets_seg/GTA5', 'list_path': '../datasets/GTA5/list',
             'image_path': '../../DATASETS/datasets_seg/GTA5/images',
             'gt_path': './datasets/GTA5/labels'},
    'synthia': {'data_root_path': './datasets/SYNTHIA', 'list_path': './datasets/SYNTHIA/list',
                'image_path': './datasets/SYNTHIA/RGB',
                'gt_path': './datasets/SYNTHIA/GT/LABELS'},
    'NTHU': {'data_root_path': './datasets/NTHU_Datasets', 'list_path': './datasets/NTHU_list'}
}


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, device=0, ignore = -1):
        super().__init__()
        self.device = device
        self.ignore = ignore
    def forward(self, output, target):
        lis = []
        for i in range(19):
            # non_zero_num = torch.nonzero(target).shape[0]
            # print(type(non_zero_num))
            gt = (target == i).float()  # B
            inter = torch.sum(gt, dim=(0, 1, 2)).cpu().numpy()  # B


            total_num = torch.prod(torch.tensor(target.shape)).float()

            k = inter.item() / total_num.item()

            lis.append(1-k)
        # print(lis)

        scaled_weight = torch.tensor(lis).cuda(self.device)
        # scaled_weight = torch.tensor([]).cuda(self.device)

        # non_zero_num = torch.nonzero(target).shape[0]
        # total_num = torch.prod(torch.tensor(target.shape)).float()
        # k = non_zero_num / total_num
        # scaled_weight = torch.tensor([k, 1-k]).cuda(self.device)

        if type(output) == list:
            loss = F.cross_entropy(output[0], target, weight=scaled_weight, ignore_index= self.ignore)
            for i in range(1, len(output)):
                loss += F.cross_entropy(output[i], target, weight=scaled_weight, ignore_index= self.ignore)
        else:
            loss = F.cross_entropy(output, target, weight=scaled_weight, ignore_index= self.ignore)

        return loss



device_ids = [0]



class Trainer():
    def __init__(self, args, cuda=None, train_id="None", logger=None):
        self.args = args
        ITER_MAX = args.each_epoch_iters
        self.device = torch.device('cuda:{}'.format(device_ids[0]))
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        self.cuda = cuda and torch.cuda.is_available()
        # self.device = torch.device('cuda' if self.cuda else 'cpu')
        self.train_id = train_id
        self.logger = logger

        self.current_MIoU = 0
        self.best_MIou = 0
        self.best_source_MIou = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.second_best_MIou = 0

        # set TensorboardX
        self.writer = SummaryWriter(self.args.checkpoint_dir)

        # Metric definition
        self.Eval = Eval(self.args.num_classes)

        # loss definition
        if args.weight_loss:
            self.loss = WeightedCrossEntropyLoss().to(self.device)
        else:
            self.loss = nn.CrossEntropyLoss(weight=None, ignore_index=-1)

        # self.loss.to(self.device)
        self.loss = self.loss.cuda(device_ids[0])

        # model
        self.model, params = get_model(self.args)
        if args.use_trained:
            self.model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, '2.pt')))
            self.model.restored = True


        # self.model.to(self.device)
        self.model = self.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        # if torch.cuda.is_available():
        #     if len(device_ids) > 1:
        #         self.model.to(torch.device('cuda:{}'.format(0)))
        #         self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        #     else:
        #         self.model.to(torch.device('cuda:{}'.format(0)))
        # self.model = nn.DataParallel(self.model, device_ids=[0,1,2,3,4,5,6])
        # self.model.to(self.device)

        if self.args.optim == "SGD":
            self.optimizer = torch.optim.SGD(
                params=params,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
            if len(device_ids) > 1:
                self.optimizer = nn.DataParallel(self.optimizer, device_ids=device_ids)

        elif self.args.optim == "Adam":
            self.optimizer = torch.optim.Adam(params, betas=(0.9, 0.99), weight_decay=self.args.weight_decay)
        # dataloader
        if self.args.dataset == "cityscapes":
            self.dataloader = City_DataLoader(self.args)
        elif self.args.dataset == "gta5":
            self.dataloader = GTA5_DataLoader(self.args)
        else:
            self.dataloader = SYNTHIA_DataLoader(self.args)
        if self.args.val_dataset == "cityscapes":
            self.test_dataloader = City_DataLoader(self.args)
        else:
            self.test_dataloader = SYNTHIA_DataLoader(self.args)
        self.dataloader.num_iterations = min(self.dataloader.num_iterations, ITER_MAX)
        print(self.args.iter_max, self.dataloader.num_iterations)
        self.epoch_num = ceil(self.args.iter_max / self.dataloader.num_iterations) if self.args.iter_stop is None else \
            ceil(self.args.iter_stop / self.dataloader.num_iterations)

    def main(self):
        # display args details
        self.logger.info("Global configuration as follows:")
        for key, val in vars(self.args).items():
            self.logger.info("{:16} {}".format(key, val))

        # choose cuda
        if self.cuda:
            current_device = torch.cuda.current_device()
            self.logger.info("This model will run on {}".format(torch.cuda.get_device_name(current_device)))
        else:
            self.logger.info("This model will run on CPU")

        # load pretrained checkpoint
        if self.args.pretrained_ckpt_file is not None:
            if os.path.isdir(self.args.pretrained_ckpt_file):
                self.args.pretrained_ckpt_file = os.path.join(self.args.checkpoint_dir, self.train_id + 'best.pth')
            self.load_checkpoint(self.args.pretrained_ckpt_file)

        if self.args.continue_training:
            self.load_checkpoint(os.path.join(self.args.checkpoint_dir, self.train_id + 'best.pth'))
            self.best_iter = self.current_iter
            self.best_source_iter = self.current_iter
        else:
            self.current_epoch = 0
        # train
        self.train()

        self.writer.close()

    def train(self):
        # self.validate() # check image summary
        pixel_num = []
        for epoch in range(self.current_epoch, self.epoch_num):

            self.train_one_epoch(pixel_num = pixel_num)
            # validate
            PA, MPA, MIoU, FWIoU = self.validate()
            self.writer.add_scalar('PA', PA, self.current_epoch)
            self.writer.add_scalar('MPA', MPA, self.current_epoch)
            self.writer.add_scalar('MIoU', MIoU, self.current_epoch)
            self.writer.add_scalar('FWIoU', FWIoU, self.current_epoch)
            #
            self.current_MIoU = MIoU
            is_best = MIoU > self.best_MIou
            torch.save(self.model.module.state_dict(),
                       os.path.join(args.checkpoint_dir, '{}.pt'.format(epoch)))


            if is_best:

                self.best_MIou = MIoU
                self.best_iter = self.current_iter
                self.logger.info("=>saving a new best checkpoint...")
                self.save_checkpoint(self.train_id + 'best.pth')
                torch.save(self.model.module.state_dict(),
                           os.path.join(args.checkpoint_dir, 'best.pt'))
            else:
                self.logger.info("=> The MIoU of val does't improve.")
                self.logger.info("=> The best MIoU of val is {} at {}".format(self.best_MIou, self.best_iter))

            self.current_epoch += 1

        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_MIou': self.current_MIoU
        }
        self.logger.info("=>best_MIou {} at {}".format(self.best_MIou, self.best_iter))
        self.logger.info(
            "=>saving the final checkpoint to " + os.path.join(self.args.checkpoint_dir, self.train_id + 'final.pth'))
        self.save_checkpoint(self.train_id + 'final.pth')

    def get_one_hot(self, label, N):
        size = list(label.size())
        label = label.view(-1)
        ones = torch.sparse.torch.eye(N).to(self.device)
        ones = ones.index_select(0, label)
        size.append(N)
        ones = ones.view(*size)
        ones = ones.transpose(2, 3)
        ones = ones.transpose(1, 2)
        return ones

    def train_one_epoch(self,pixel_num):
        tqdm_epoch = tqdm(self.dataloader.data_loader,
                          total=self.dataloader.num_iterations,
                          desc="Train Epoch-{}-total-{}".format(self.current_epoch + 1, self.epoch_num),file=sys.stdout)
        self.logger.info("Training one epoch...")
        self.Eval.reset()

        train_loss = []
        loss_seg_value_2 = 0
        iter_num = self.dataloader.num_iterations

        if self.args.freeze_bn:
            self.model.eval()
            self.logger.info("freeze bacth normalization successfully!")
        else:
            self.model.train()
        # Initialize your average meters

        batch_idx = 0

        for x, y, _ in tqdm_epoch:
            self.poly_lr_scheduler(
                optimizer=self.optimizer,
                init_lr=self.args.lr,
                iter=self.current_iter,
                max_iter=self.args.iter_max,
                power=self.args.poly_power,
            )
            if self.args.iter_stop is not None and self.current_iter >= self.args.iter_stop:
                self.logger.info(
                    "iteration arrive {}(early stop)/{}(total step)!".format(self.args.iter_stop, self.args.iter_max))
                break
            if self.current_iter >= self.args.iter_max:
                self.logger.info("iteration arrive {}!".format(self.args.iter_max))
                break
            if len(device_ids) > 1:
                self.writer.add_scalar('learning_rate', self.optimizer.module.param_groups[0]["lr"], self.current_iter)
            else:
                self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]["lr"], self.current_iter)

            if self.cuda:
                x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)
            y = torch.squeeze(y, 1)
            self.optimizer.zero_grad()

            # model
            pred = self.model(x)


            weights_keys = self.model.state_dict().keys()

            selected_keys_classify_1 = []
            selected_keys_classify_2 = []
            for key in weights_keys:
                if "classifier_1.weight" in key:
                    selected_keys_classify_1.append(key)
                if "classifier_2.weight" in key:
                    selected_keys_classify_2.append(key)

            for key in selected_keys_classify_1:
                if "num_batches_tracked" in key:
                    continue
                weights_t = self.model.state_dict()[key]
            classsifier_1_weights = weights_t.squeeze()
            for key in selected_keys_classify_2:
                if "num_batches_tracked" in key:
                    continue
                weights_t = self.model.state_dict()[key]
            classsifier_2_weights = weights_t.squeeze()
            # print(classsifier_1_weights.size(),classsifier_2_weights.size())




            if isinstance(pred, tuple):

                pred_2 = pred[1]
                pred_lay2 = pred[2]
                pred_lay1 = pred[3]
                mid_lay2_ori = pred[4]
                mid_lay2_ined = pred[5]
                mid_lay1_ori = pred[6]
                mid_lay1_ined = pred[7]
                mid_lay2_iwed = pred[8]
                mid_lay1_iwed = pred[9]
                pred = pred[0]

            a_ = torch.ones(y.size()[0],y.size()[1],y.size()[2],dtype=torch.long)*args.num_classes
            a_ = a_.to(self.device)
            y_ = torch.where(y==-1, a_, y).to(self.device)
            gt_one_hot = self.get_one_hot(y_, args.num_classes+1).to(self.device)

            outs_lay2 = []
            for i in args.selected_classes:
                mask = torch.unsqueeze(gt_one_hot[:, i, :, :], 1)
                mask = F.interpolate(mask, size=mid_lay2_ori.size()[2:],mode='nearest')
                out = mid_lay2_ori * mask
                out = self.model.module.SAN_stage_2.IN(out)
                # out = nn.InstanceNorm2d(512, affine=True)(out)
                outs_lay2.append(out)
            mid_lay2_label = sum(outs_lay2)
            mid_lay2_label = self.model.module.SAN_stage_2.relu(mid_lay2_label)

            outs_lay1 = []
            for i in args.selected_classes:
                mask = torch.unsqueeze(gt_one_hot[:, i, :, :], 1)
                mask = F.interpolate(mask, size=mid_lay1_ori.size()[2:], mode='nearest')
                out = mid_lay1_ori * mask
                out = self.model.module.SAN_stage_1.IN(out)
                # out = nn.InstanceNorm2d(512, affine=True)(out)
                outs_lay1.append(out)
            mid_lay1_label = sum(outs_lay1)
            mid_lay1_label = self.model.module.SAN_stage_1.relu(mid_lay1_label)


            # loss
            loss_main = self.loss(pred, y)
            loss_lay2 = 0.1 * self.loss(pred_lay2, y)
            loss_lay1 = 0.1 * self.loss(pred_lay1, y)
            loss_in_lay2 = 0.1 * F.smooth_l1_loss(mid_lay2_ined, mid_lay2_label)
            loss_in_lay1 = 0.1 * F.smooth_l1_loss(mid_lay1_ined, mid_lay1_label)
            loss_iw_lay2 = 0.1 * mid_lay2_iwed
            loss_iw_lay1 = 0.1 * mid_lay1_iwed
            cur_loss = loss_main + loss_lay2 + loss_lay1 + loss_in_lay2 + loss_in_lay1 + loss_iw_lay2 + loss_iw_lay1


            #########################
            lis = []
            for i in range(19):
                # non_zero_num = torch.nonzero(target).shape[0]
                # print(type(non_zero_num))
                gt = (y == i).float()  # B
                inter = torch.sum(gt, dim=(0, 1, 2)).cpu().numpy()  # B

                total_num = torch.prod(torch.tensor(y.shape)).float()

                k = inter.item() / total_num.item()

                lis.append(k)
            pixel_num.append(lis)
            #########################


            if self.args.multi:
                loss_2 = self.args.lambda_seg * self.loss(pred_2, y)
                cur_loss += loss_2
                loss_seg_value_2 += loss_2.cpu().item() / iter_num

            tqdm_epoch.set_postfix(loss_total=cur_loss.item(), loss_main=loss_main.item())

            # optimizer
            cur_loss.backward()
            if len(device_ids) > 1:
                self.optimizer.module.step()
            else:
                self.optimizer.step()

            train_loss.append(cur_loss.item())

            # if batch_idx % 50 == 0:
            #     if self.args.multi:
            #         self.logger.info("The train loss of epoch{}-batch-{}:{};{}".format(self.current_epoch,
            #                                                                            batch_idx, cur_loss.item(),
            #                                                                            loss_2.item()))
            #     else:
            #         self.logger.info("The train loss of epoch{}-batch-{}:{}".format(self.current_epoch,
            #                                                                         batch_idx, cur_loss.item()))

            batch_idx += 1

            self.current_iter += 1

            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')

            pred = pred.data.cpu().numpy()
            label = y.cpu().numpy()
            argpred = np.argmax(pred, axis=1)
            self.Eval.add_batch(label, argpred)

            if batch_idx == self.dataloader.num_iterations:
                break

        #######
        mean = np.mean(pixel_num, axis=0)
        sorted_id = sorted(range(len(mean)), key=lambda k: mean[k], reverse=True)
        print(sorted_id)
        ########

        self.log_one_train_epoch(x, label, argpred, train_loss)
        tqdm_epoch.close()

    def log_one_train_epoch(self, x, label, argpred, train_loss):
        # show train image on tensorboard
        images_inv = inv_preprocess(x.clone().cpu(), self.args.show_num_images,
                                    numpy_transform=self.args.numpy_transform)
        labels_colors = decode_labels(label, self.args.show_num_images)
        preds_colors = decode_labels(argpred, self.args.show_num_images)
        for index, (img, lab, color_pred) in enumerate(zip(images_inv, labels_colors, preds_colors)):
            self.writer.add_image('train/' + str(index) + '/Images', img, self.current_epoch)
            self.writer.add_image('train/' + str(index) + '/Labels', lab, self.current_epoch)
            self.writer.add_image('train/' + str(index) + '/preds', color_pred, self.current_epoch)

        if self.args.class_16:
            PA = self.Eval.Pixel_Accuracy()
            MPA_16, MPA = self.Eval.Mean_Pixel_Accuracy()
            MIoU_16, MIoU = self.Eval.Mean_Intersection_over_Union()
            FWIoU_16, FWIoU = self.Eval.Frequency_Weighted_Intersection_over_Union()
        else:
            PA = self.Eval.Pixel_Accuracy()
            MPA = self.Eval.Mean_Pixel_Accuracy()
            MIoU = self.Eval.Mean_Intersection_over_Union()
            FWIoU = self.Eval.Frequency_Weighted_Intersection_over_Union()

        self.logger.info('\nEpoch:{}, train PA1:{}, MPA1:{}, MIoU1:{}, FWIoU1:{}'.format(self.current_epoch, PA, MPA,
                                                                                         MIoU, FWIoU))
        self.writer.add_scalar('train_PA', PA, self.current_epoch)
        self.writer.add_scalar('train_MPA', MPA, self.current_epoch)
        self.writer.add_scalar('train_MIoU', MIoU, self.current_epoch)
        self.writer.add_scalar('train_FWIoU', FWIoU, self.current_epoch)

        tr_loss = sum(train_loss) / len(train_loss) if isinstance(train_loss, list) else train_loss
        self.writer.add_scalar('train_loss', tr_loss, self.current_epoch)
        tqdm.write("The average loss of train epoch-{}-:{}".format(self.current_epoch, tr_loss))

    def validate(self, mode='val'):
        self.logger.info('\nvalidating one epoch...')
        self.Eval.reset()
        with torch.no_grad():
            tqdm_batch = tqdm(self.test_dataloader.val_loader, total=self.test_dataloader.valid_iterations,
                              desc="Val Epoch-{}-".format(self.current_epoch + 1),file=sys.stdout)
            if mode == 'val':
                self.model.eval()

            i = 0

            for x, y, id in tqdm_batch:
                if self.cuda:
                    x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)

                # model
                pred = self.model(x)
                if isinstance(pred, tuple):
                    pred_2 = pred[1]
                    pred = pred[0]
                    pred_P = F.softmax(pred, dim=1)
                    pred_P_2 = F.softmax(pred_2, dim=1)
                y = torch.squeeze(y, 1)

                pred = pred.data.cpu().numpy()
                label = y.cpu().numpy()
                argpred = np.argmax(pred, axis=1)

                self.Eval.add_batch(label, argpred)

            # show val result on tensorboard
            images_inv = inv_preprocess(x.clone().cpu(), self.args.show_num_images,
                                        numpy_transform=self.args.numpy_transform)
            labels_colors = decode_labels(label, self.args.show_num_images)
            preds_colors = decode_labels(argpred, self.args.show_num_images)
            for index, (img, lab, color_pred) in enumerate(zip(images_inv, labels_colors, preds_colors)):
                self.writer.add_image(str(index) + '/Images', img, self.current_epoch)
                self.writer.add_image(str(index) + '/Labels', lab, self.current_epoch)
                self.writer.add_image(str(index) + '/preds', color_pred, self.current_epoch)

            if self.args.class_16:
                def val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA_16, MPA_13 = Eval.Mean_Pixel_Accuracy()
                    MIoU_16, MIoU_13 = Eval.Mean_Intersection_over_Union()
                    FWIoU_16, FWIoU_13 = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC_16, PC_13 = Eval.Mean_Precision()
                    print("########## Eval{} ############".format(name))

                    self.logger.info(
                        '\nEpoch:{:.3f}, {} PA:{:.3f}, MPA_16:{:.3f}, MIoU_16:{:.3f}, FWIoU_16:{:.3f}, PC_16:{:.3f}'.format(
                            self.current_epoch, name, PA, MPA_16,
                            MIoU_16, FWIoU_16, PC_16))
                    self.logger.info(
                        '\nEpoch:{:.3f}, {} PA:{:.3f}, MPA_13:{:.3f}, MIoU_13:{:.3f}, FWIoU_13:{:.3f}, PC_13:{:.3f}'.format(
                            self.current_epoch, name, PA, MPA_13,
                            MIoU_13, FWIoU_13, PC_13))
                    self.writer.add_scalar('PA' + name, PA, self.current_epoch)
                    self.writer.add_scalar('MPA_16' + name, MPA_16, self.current_epoch)
                    self.writer.add_scalar('MIoU_16' + name, MIoU_16, self.current_epoch)
                    self.writer.add_scalar('FWIoU_16' + name, FWIoU_16, self.current_epoch)
                    self.writer.add_scalar('MPA_13' + name, MPA_13, self.current_epoch)
                    self.writer.add_scalar('MIoU_13' + name, MIoU_13, self.current_epoch)
                    self.writer.add_scalar('FWIoU_13' + name, FWIoU_13, self.current_epoch)
                    return PA, MPA_13, MIoU_13, FWIoU_13
            else:
                def val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA = Eval.Mean_Pixel_Accuracy()
                    MIoU = Eval.Mean_Intersection_over_Union()
                    FWIoU = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC = Eval.Mean_Precision()
                    print("########## Eval{} ############".format(name))

                    self.logger.info(
                        '\nEpoch:{:.3f}, {} PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(
                            self.current_epoch, name, PA, MPA,
                            MIoU, FWIoU, PC))
                    self.writer.add_scalar('PA' + name, PA, self.current_epoch)
                    self.writer.add_scalar('MPA' + name, MPA, self.current_epoch)
                    self.writer.add_scalar('MIoU' + name, MIoU, self.current_epoch)
                    self.writer.add_scalar('FWIoU' + name, FWIoU, self.current_epoch)
                    return PA, MPA, MIoU, FWIoU

            PA, MPA, MIoU, FWIoU = val_info(self.Eval, "")
            tqdm_batch.close()

        return PA, MPA, MIoU, FWIoU

    def validate_source(self):
        self.logger.info('\nvalidating source domain...')
        self.Eval.reset()
        with torch.no_grad():
            tqdm_batch = tqdm(self.source_val_dataloader, total=self.dataloader.valid_iterations,
                              desc="Source Val Epoch-{}-".format(self.current_epoch + 1))
            self.model.eval()
            i = 0
            for x, y, id in tqdm_batch:
                # y.to(torch.long)
                if self.cuda:
                    x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)

                # model
                pred = self.model(x)

                if isinstance(pred, tuple):
                    pred_2 = pred[1]
                    pred = pred[0]
                    pred_P = F.softmax(pred, dim=1)
                    pred_P_2 = F.softmax(pred_2, dim=1)
                y = torch.squeeze(y, 1)

                pred = pred.data.cpu().numpy()
                label = y.cpu().numpy()
                argpred = np.argmax(pred, axis=1)

                self.Eval.add_batch(label, argpred)

                i += 1
                if i == self.dataloader.valid_iterations:
                    break

            # show val result on tensorboard
            images_inv = inv_preprocess(x.clone().cpu(), self.args.show_num_images,
                                        numpy_transform=self.args.numpy_transform)
            labels_colors = decode_labels(label, self.args.show_num_images)
            preds_colors = decode_labels(argpred, self.args.show_num_images)
            for index, (img, lab, color_pred) in enumerate(zip(images_inv, labels_colors, preds_colors)):
                self.writer.add_image('source_eval/' + str(index) + '/Images', img, self.current_epoch)
                self.writer.add_image('source_eval/' + str(index) + '/Labels', lab, self.current_epoch)
                self.writer.add_image('source_eval/' + str(index) + '/preds', color_pred, self.current_epoch)

            if self.args.class_16:
                def source_val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA_16, MPA_13 = Eval.Mean_Pixel_Accuracy()
                    MIoU_16, MIoU_13 = Eval.Mean_Intersection_over_Union()
                    FWIoU_16, FWIoU_13 = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC_16, PC_13 = Eval.Mean_Precision()
                    print("########## Source Eval{} ############".format(name))

                    self.logger.info(
                        '\nEpoch:{:.3f}, source {} PA:{:.3f}, MPA_16:{:.3f}, MIoU_16:{:.3f}, FWIoU_16:{:.3f}, PC_16:{:.3f}'.format(
                            self.current_epoch, name, PA, MPA_16,
                            MIoU_16, FWIoU_16, PC_16))
                    self.logger.info(
                        '\nEpoch:{:.3f}, source {} PA:{:.3f}, MPA_13:{:.3f}, MIoU_13:{:.3f}, FWIoU_13:{:.3f}, PC_13:{:.3f}'.format(
                            self.current_epoch, name, PA, MPA_13,
                            MIoU_13, FWIoU_13, PC_13))
                    self.writer.add_scalar('source_PA' + name, PA, self.current_epoch)
                    self.writer.add_scalar('source_MPA_16' + name, MPA_16, self.current_epoch)
                    self.writer.add_scalar('source_MIoU_16' + name, MIoU_16, self.current_epoch)
                    self.writer.add_scalar('source_FWIoU_16' + name, FWIoU_16, self.current_epoch)
                    self.writer.add_scalar('source_MPA_13' + name, MPA_13, self.current_epoch)
                    self.writer.add_scalar('source_MIoU_13' + name, MIoU_13, self.current_epoch)
                    self.writer.add_scalar('source_FWIoU_13' + name, FWIoU_13, self.current_epoch)
                    return PA, MPA_13, MIoU_13, FWIoU_13
            else:
                def source_val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA = Eval.Mean_Pixel_Accuracy()
                    MIoU = Eval.Mean_Intersection_over_Union()
                    FWIoU = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC = Eval.Mean_Precision()

                    self.writer.add_scalar('source_PA' + name, PA, self.current_epoch)
                    self.writer.add_scalar('source_MPA' + name, MPA, self.current_epoch)
                    self.writer.add_scalar('source_MIoU' + name, MIoU, self.current_epoch)
                    self.writer.add_scalar('source_FWIoU' + name, FWIoU, self.current_epoch)
                    print("########## Source Eval{} ############".format(name))

                    self.logger.info(
                        '\nEpoch:{:.3f}, source {} PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(
                            self.current_epoch, name, PA, MPA,
                            MIoU, FWIoU, PC))
                    return PA, MPA, MIoU, FWIoU

            PA, MPA, MIoU, FWIoU = source_val_info(self.Eval, "")
            tqdm_batch.close()

        is_best = MIoU > self.best_source_MIou
        if is_best:
            self.best_source_MIou = MIoU
            self.best_source_iter = self.current_iter
            self.logger.info("=>saving a new best source checkpoint...")
            self.save_checkpoint(self.train_id + 'source_best.pth')
        else:
            self.logger.info("=> The source MIoU of val does't improve.")
            self.logger.info(
                "=> The best source MIoU of val is {} at {}".format(self.best_source_MIou, self.best_source_iter))

        return PA, MPA, MIoU, FWIoU

    def save_checkpoint(self, filename=None):
        """
        Save checkpoint if a new best is achieved
        :param state:
        :param is_best:
        :param filepath:
        :return:
        """
        filename = os.path.join(self.args.checkpoint_dir, filename)
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_MIou': self.best_MIou
        }
        torch.save(state, filename)

    def load_checkpoint(self, filename):
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.module.load_state_dict(checkpoint)
            self.logger.info("Checkpoint loaded successfully from " + filename)
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.args.checkpoint_dir))
            self.logger.info("**First time to train**")

    def poly_lr_scheduler(self, optimizer, init_lr=None, iter=None,
                          max_iter=None, power=None):
        init_lr = self.args.lr if init_lr is None else init_lr
        iter = self.current_iter if iter is None else iter
        max_iter = self.args.iter_max if max_iter is None else max_iter
        power = self.args.poly_power if power is None else power
        new_lr = init_lr * (1 - float(iter) / max_iter) ** power
        if len(device_ids) > 1:
            optimizer.module.param_groups[0]["lr"] = new_lr
            if len(optimizer.module.param_groups) == 2:
                optimizer.module.param_groups[1]["lr"] = 10 * new_lr
        else:
            optimizer.param_groups[0]["lr"] = new_lr
            if len(optimizer.param_groups) == 2:
                optimizer.param_groups[1]["lr"] = 10 * new_lr


def add_train_args(arg_parser):
    # Path related arguments
    arg_parser.add_argument('--data_root_path', type=str, default='../../DATASETS/datasets_seg/GTA5',
                            help="the root path of dataset")
    arg_parser.add_argument('--list_path', type=str, default='../datasets/GTA_640/list',
                            help="the root path of dataset")
    arg_parser.add_argument('--checkpoint_dir', default="./log/gta5_pretrain_2",
                            help="the path of ckpt file")
    arg_parser.add_argument('--xuanran_path', default=None,
                            help="the path of ckpt file")

    # Model related arguments
    arg_parser.add_argument('--weight_loss', default=True,
                            help="if use weight loss")
    arg_parser.add_argument('--use_trained', default=False,
                            help="if use trained model")
    arg_parser.add_argument('--backbone', default='Deeplab50_CLASS_INW',
                            help="backbone of encoder")
    arg_parser.add_argument('--bn_momentum', type=float, default=0.1,
                            help="batch normalization momentum")
    arg_parser.add_argument('--imagenet_pretrained', type=str2bool, default=True,
                            help="whether apply imagenet pretrained weights")
    arg_parser.add_argument('--pretrained_ckpt_file', type=str, default=None,
                            help="whether apply pretrained checkpoint")
    arg_parser.add_argument('--continue_training', type=str2bool, default=False,
                            help="whether to continue training ")
    arg_parser.add_argument('--show_num_images', type=int, default=2,
                            help="show how many images during validate")

    # train related arguments
    arg_parser.add_argument('--seed', default=12345, type=int,
                            help='random seed')
    arg_parser.add_argument('--gpu', type=str, default="0",
                            help=" the num of gpu")
    arg_parser.add_argument('--batch_size_per_gpu', default=2, type=int,
                            help='input batch size')
    arg_parser.add_argument('--alpha', default=0.3, type=int,
                            help='input mix alpha')

    # dataset related arguments
    arg_parser.add_argument('--dataset', default='gta5', type=str,
                            help='dataset choice')
    arg_parser.add_argument('--val_dataset', type=str, default='cityscapes',
                        help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')
    arg_parser.add_argument('--base_size', default="640,640", type=str,
                            help='crop size of image')
    arg_parser.add_argument('--crop_size', default="640,640", type=str,
                            help='base size of image')
    arg_parser.add_argument('--target_base_size', default="1024,512", type=str,
                            help='crop size of target image')
    arg_parser.add_argument('--target_crop_size', default="1024,512", type=str,
                            help='base size of target image')
    arg_parser.add_argument('--num_classes', default=19, type=int,
                            help='num class of mask')
    arg_parser.add_argument('--data_loader_workers', default=1, type=int,
                            help='num_workers of Dataloader')
    arg_parser.add_argument('--pin_memory', default=2, type=int,
                            help='pin_memory of Dataloader')
    arg_parser.add_argument('--split', type=str, default='train',
                            help="choose from train/val/test/trainval/all")
    arg_parser.add_argument('--random_mirror', default=True, type=str2bool,
                            help='add random_mirror')
    arg_parser.add_argument('--random_crop', default=False, type=str2bool,
                            help='add random_crop')
    arg_parser.add_argument('--resize', default=True, type=str2bool,
                            help='resize')
    arg_parser.add_argument('--gaussian_blur', default=True, type=str2bool,
                            help='add gaussian_blur')
    arg_parser.add_argument('--numpy_transform', default=True, type=str2bool,
                            help='image transform with numpy style')
    arg_parser.add_argument('--color_jitter', default=True, type=str2bool,
                            help='image transform with numpy style')

    # optimization related arguments

    arg_parser.add_argument('--freeze_bn', type=str2bool, default=False,
                            help="whether freeze BatchNormalization")
    arg_parser.add_argument('--optim', default="SGD", type=str,
                            help='optimizer')
    arg_parser.add_argument('--momentum', type=float, default=0.9)
    arg_parser.add_argument('--weight_decay', type=float, default=5e-4)

    arg_parser.add_argument('--lr', type=float, default=5e-4,
                            help="init learning rate ")
    arg_parser.add_argument('--iter_max', type=int, default=200000,
                            help="the maxinum of iteration")
    arg_parser.add_argument('--iter_stop', type=int, default=200000,
                            help="the early stop step")
    arg_parser.add_argument('--each_epoch_iters', default=1000,
                            help="the path of ckpt file")
    arg_parser.add_argument('--poly_power', type=float, default=0.9,
                            help="poly_power")
    arg_parser.add_argument('--selected_classes', default=[0,10,2,1,8],
                            help="poly_power")

    # multi-level output

    arg_parser.add_argument('--multi', default=False, type=str2bool,
                            help='output model middle feature')
    arg_parser.add_argument('--lambda_seg', type=float, default=0.1,
                            help="lambda_seg of middle output")
    return arg_parser


def init_args(args):
    # args.batch_size = args.batch_size_per_gpu * ceil(len(args.gpu) / 2)
    args.batch_size = args.batch_size_per_gpu
    print("batch size: ", args.batch_size)

    train_id = str(args.dataset)

    crop_size = args.crop_size.split(',')
    base_size = args.base_size.split(',')
    if len(crop_size) == 1:
        args.crop_size = int(crop_size[0])
        args.base_size = int(base_size[0])
    else:
        args.crop_size = (int(crop_size[0]), int(crop_size[1]))
        args.base_size = (int(base_size[0]), int(base_size[1]))


    target_crop_size = args.target_crop_size.split(',')
    target_base_size = args.target_base_size.split(',')
    if len(target_crop_size) == 1:
        args.target_crop_size = int(target_crop_size[0])
        args.target_base_size = int(target_base_size[0])
    else:
        args.target_crop_size = (int(target_crop_size[0]), int(target_crop_size[1]))
        args.target_base_size = (int(target_base_size[0]), int(target_base_size[1]))

    # if not args.continue_training:
    #     if os.path.exists(args.checkpoint_dir):
    #         print("checkpoint dir exists, which will be removed")
    #         import shutil
    #         shutil.rmtree(args.checkpoint_dir, ignore_errors=True)
    #     os.mkdir(args.checkpoint_dir)

    if args.data_root_path is None:
        args.data_root_path = datasets_path[args.dataset]['data_root_path']
        args.list_path = datasets_path[args.dataset]['list_path']
        args.image_filepath = datasets_path[args.dataset]['image_path']
        args.gt_filepath = datasets_path[args.dataset]['gt_path']

    args.class_16 = True if args.num_classes == 16 else False
    args.class_13 = True if args.num_classes == 13 else False

    # logger configure
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(args.checkpoint_dir, 'train_log.txt'))
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    return args, train_id, logger


if __name__ == '__main__':
    print(torch.cuda.is_available())
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), 'PyTorch>=1.0.0 is required'
    warnings.filterwarnings('ignore')
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_train_args(arg_parser)

    args = arg_parser.parse_args()
    args, train_id, logger = init_args(args)

    agent = Trainer(args=args, cuda=True, train_id=train_id, logger=logger)
    agent.main()
