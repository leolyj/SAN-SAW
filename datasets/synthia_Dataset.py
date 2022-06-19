 # -*- coding: utf-8 -*-
import random
import scipy.io
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import copy
import os
import torch
import torch.utils.data as data
import torchvision.transforms as ttransforms
import imageio

from datasets.cityscapes_Dataset import City_Dataset, City_DataLoader

imageio.plugins.freeimage.download()

class SYNTHIA_Dataset(City_Dataset):
    def __init__(self,
                 args,
                 data_root_path='./datasets/SYNTHIA',
                 list_path='./datasets/SYNTHIA/list',
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True,
                 class_16=False):

        self.args = args
        self.data_path=data_root_path
        self.list_path=list_path
        self.split=split
        self.base_size=base_size
        self.crop_size=crop_size

        self.base_size = self.base_size if isinstance(self.base_size, tuple) else (self.base_size, self.base_size)
        self.crop_size = self.crop_size if isinstance(self.crop_size, tuple) else (self.crop_size, self.crop_size)
        self.training = training

        self.random_mirror = args.random_mirror
        self.random_crop = args.random_crop
        self.resize = args.resize
        self.gaussian_blur = args.gaussian_blur

        item_list_filepath = os.path.join(self.list_path, self.split+".txt")

        if not os.path.exists(item_list_filepath):
            raise Warning("split must be train/val/trainavl/test")

        self.image_filepath = os.path.join(self.data_path, "RGB")

        self.gt_filepath = os.path.join(self.data_path, "GT/LABELS")

        self.items = [id.strip() for id in open(item_list_filepath)]

        ignore_label = -1
        self.id_to_trainid = {1:10, 2:2, 3:0, 4:1, 5:4, 6:8, 7:5, 8:13, 
                            9:7, 10:11, 11:18, 12:17, 15:6, 16:9, 17:12, 
                            18:14, 19:15, 20:16, 21:3}
        # only consider 16 shared classes
        self.class_16 = class_16
        synthia_set_16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
        self.trainid_to_16id = {id:i for i,id in enumerate(synthia_set_16)}
        self.class_13 = False

        print("{} num images in GTA5 {} set have been loaded.".format(len(self.items), self.split))

    def __getitem__(self, item):
        id = int(self.items[item])

        image_path = os.path.join(self.image_filepath, "{:0>7d}.png".format(id))
        image = Image.open(image_path).convert("RGB")

        gt_image_path = os.path.join(self.gt_filepath, "{:0>7d}.png".format(id))
        gt_image = imageio.imread(gt_image_path, format='PNG-FI')[:,:,0]
        gt_image = Image.fromarray(np.uint8(gt_image))

        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.training:
            image, gt_image = self._train_sync_transform(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform(image, gt_image)

        return image, gt_image, item

class SYNTHIA_DataLoader():
    def __init__(self, args, training=True):

        self.args = args

        data_set = SYNTHIA_Dataset(args, 
                                data_root_path=args.data_root_path,
                                list_path=args.list_path,
                                split=args.split,
                                base_size=args.base_size,
                                crop_size=args.crop_size,
                                training=training)

        if self.args.split == "train" or self.args.split == "trainval" or self.args.split =="all":
            self.data_loader = data.DataLoader(data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=True,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)
        elif self.args.split =="val" or self.args.split == "test":
            self.data_loader = data.DataLoader(data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=False,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)
        else:
            raise Warning("split must be train/val/trainavl/test/all")

        val_split = 'val' if self.args.split == "train" else 'test'
        val_set = SYNTHIA_Dataset(args, 
                            data_root_path=args.data_root_path,
                            list_path=args.list_path,
                            split=val_split,
                            base_size=args.base_size,
                            crop_size=args.crop_size,
                            training=False)
        self.val_loader = data.DataLoader(val_set,
                                            batch_size=self.args.batch_size,
                                            shuffle=False,
                                            num_workers=self.args.data_loader_workers,
                                            pin_memory=self.args.pin_memory,
                                            drop_last=True)
        self.valid_iterations = (len(val_set) + self.args.batch_size) // self.args.batch_size

        self.num_iterations = (len(data_set) + self.args.batch_size) // self.args.batch_size