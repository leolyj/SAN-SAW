# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
affine_par = True
import torch.utils.model_zoo as model_zoo
from graphs.models.SAW import SAW
from graphs.models.SAN import SAN


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        # self.IN = None
        # if IN:
        #     self.IN = nn.InstanceNorm2d(planes*4, affine=affine_par)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # if self.IN is not None:
        #
        #     out = self.IN(out)
        out = self.relu(out)

        return out


class class_in_block(nn.Module):

    def __init__(self, inplanes, classin_classes=None):
        super(class_in_block, self).__init__()

        self.IN = nn.InstanceNorm2d(inplanes, affine=affine_par)
        self.classin_classes = classin_classes
        self.branches = nn.ModuleList()
        for i in classin_classes:
            self.branches.append(
                nn.Conv2d(3, 1, kernel_size=7, stride=1, padding=3, bias=False))

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, masks):
        outs=[]
        idx = 0
        masks = F.softmax(masks,dim=1)
        for i in self.classin_classes:
            mask = torch.unsqueeze(masks[:,i,:,:],1)
            mid = x * mask
            avg_out = torch.mean(mid, dim=1, keepdim=True)
            max_out,_ = torch.max(mid,dim=1, keepdim=True)
            atten = torch.cat([avg_out,max_out,mask],dim=1)
            atten = self.sigmoid(self.branches[idx](atten))
            out = mid*atten
            out = self.IN(out)
            outs.append(out)
        out_ = sum(outs)
        out_ = self.relu(out_)

        return out_


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out


class ResNetMulti(nn.Module):
    def __init__(self,args, block, layers, num_classes):
        self.inplanes = 64
        super(ResNetMulti, self).__init__()

        self.classifier_1 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=True)
        self.classifier_2 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, bias=True)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        self.in1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, 1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        # self.layer1_class_in = class_in_block(inplanes=256, classin_classes=args.selected_classes)
        # self.layer2_class_in = class_in_block(inplanes=512, classin_classes=args.selected_classes)
        self.SAN_stage_1 = SAN(inplanes=256, selected_classes=args.selected_classes)
        self.SAN_stage_2 = SAN(inplanes=512, selected_classes=args.selected_classes)
        self.SAW_stage_1 = SAW(args, dim=256, relax_denom=2.0, classifier=self.classifier_1)
        self.SAW_stage_2 = SAW(args, dim=512, relax_denom=2.0, classifier=self.classifier_2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def print_weights(self):
        weights_keys = self.layer1_pred.state_dict().keys()
        for key in weights_keys:
            if "num_batches_tracked" in key:
                continue
            weights_t = self.layer1_pred.state_dict()[key].numpy()
        return weights_t

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x):

        input_size = x.size()[2:]
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x_1_ori = x
        x1 = self.classifier_1(x.detach())
        x = self.SAN_stage_1(x,x1)
        x_1_ined = x

        saw_loss_lay1 = self.SAW_stage_1(x)
        x1 = F.interpolate(x1, size=input_size, mode='bilinear', align_corners=True)

        x = self.layer2(x)
        x_2_ori = x
        x2 = self.classifier_2(x.detach())
        x = self.SAN_stage_2(x, x2)
        x_2_ined = x

        saw_loss_lay2 = self.SAW_stage_2(x)
        x2 = F.interpolate(x2, size=input_size, mode='bilinear', align_corners=True)

        x = self.layer3(x)
        x3 = self.layer5(x)
        x3 = F.interpolate(x3, size=input_size, mode='bilinear', align_corners=True)

        x4 = self.layer4(x)
        x4 = self.layer6(x4)
        x4 = F.interpolate(x4, size=input_size, mode='bilinear', align_corners=True)

        return x4, x3, x2, x1, x_2_ori, x_2_ined, x_1_ori, x_1_ined, saw_loss_lay2, saw_loss_lay1

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.in1)
        b.append(self.layer1)
        b.append(self.SAN_stage_1)
        b.append(self.layer2)
        b.append(self.SAN_stage_2)
        b.append(self.layer3)
        b.append(self.layer4)
        b.append(self.classifier_1)
        b.append(self.classifier_2)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.lr},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.lr}]


def Res50_ClassINW(args,num_classes=21, pretrained=True):
    model = ResNetMulti(args, Bottleneck, [3, 4, 6, 3], num_classes)

    if pretrained:


        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']),
                              strict=False)

        # restore_from = './pretrained_model/DeepLab_resnet_pretrained_init-f81d91e8.pth'
        # # restore_from = './pretrained_model/GTA5_source.pth'
        # saved_state_dict = torch.load(restore_from)
        #
        # new_params = model.state_dict().copy()
        # for i in saved_state_dict:
        #     i_parts = i.split('.')
        #     if not i_parts[1] == 'layer5':
        #         new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        # model.load_state_dict(new_params)
    return model

if __name__ == '__main__':
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], 19)
    restore_from = './pretrained_model/DeepLab_resnet_pretrained_init-f81d91e8.pth'
    # restore_from = './pretrained_model/GTA5_source.pth'
    saved_state_dict = torch.load(restore_from)
    # for i in saved_state_dict:
    #     print("i:",i)
    #     i_parts = i.split('.')
    #     print(i_parts[0],i_parts[1])

    new_params = model.state_dict().copy()

    for i in saved_state_dict:
        print("i:",i)
        i_parts = i.split('.')


    for i in new_params:
        print("i_new:",i)
        i_parts = i.split('.')

    for i in saved_state_dict:
        i_parts = i.split('.')
        if not i_parts[1] == 'layer5':
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]

    model.load_state_dict(new_params)

