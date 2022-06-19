# -*- coding: utf-8 -*-
import random
import scipy.io
from PIL import Image, ImageOps, ImageFilter, ImageFile
import numpy as np
import copy
import os
import torch
import torch.utils.data as data
import torchvision.transforms as ttransforms

from datasets.cityscapes_Dataset import City_Dataset, City_DataLoader, Beiyong_Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class GTA5_Dataset(City_Dataset):
    def __init__(self,
                 args,
                 data_root_path='../datasets/GTA5',
                 list_path='../datasets/GTA5/list',
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True):
        self.xuanran = args.xuanran_path
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
        self.color_jitter = args.color_jitter

        item_list_filepath = os.path.join(self.list_path, self.split+".txt")

        if not os.path.exists(item_list_filepath):
            raise Warning("split must be train/val/trainval/test/all")

        self.image_filepath = os.path.join(self.data_path, "images")

        self.gt_filepath = os.path.join(self.data_path, "labels")

        # self.items = [int(id.strip()) for id in open(item_list_filepath)]
        self.items = [id.strip() for id in open(item_list_filepath)]


        ignore_label = -1
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        self.class_16 = False
        self.class_13 = False

        print("{} num images in GTA5 {} set have been loaded.".format(len(self.items), self.split))

    def __getitem__(self, item):
        id = self.items[item]


        # image_path = os.path.join(self.image_filepath, "{0:05d}.png".format(id))
        image_path = os.path.join(self.image_filepath, "{}.png".format(id))

        image = Image.open(image_path).convert("RGB")

        # gt_image_path = os.path.join(self.gt_filepath, "{0:05d}.png".format(id))
        gt_image_path = os.path.join(self.gt_filepath, "{}.png".format(id))

        gt_image = Image.open(gt_image_path)

        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.training:
            image, gt_image = self._train_sync_transform(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform(image, gt_image)

        return image, gt_image, item

class GTA5_DataLoader():
    def __init__(self, args, training=True):

        self.args = args

        data_set = GTA5_Dataset(args,
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
        val_set = GTA5_Dataset(args,
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



class GTA5_xuanran_Dataset(City_Dataset):
    def __init__(self,
                 args,
                 data_root_path='../datasets/GTA5',
                 list_path='../datasets/GTA5/list',
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True):
        self.xuanran = args.xuanran_path

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
            raise Warning("split must be train/val/trainval/test/all")

        self.image_filepath = os.path.join(self.data_path, "images")

        self.gt_filepath = os.path.join(self.data_path, "labels")

        # self.items = [int(id.strip()) for id in open(item_list_filepath)]
        self.items = [id.strip() for id in open(item_list_filepath)]
        self.images_lst = []
        for id in self.items:
            image_path = os.path.join(self.image_filepath, "{}.png".format(id))
            self.images_lst.append(image_path)


        if self.xuanran is not None:
            self.xuanran_lst = []



            for id in self.items:
                element = []
                for i in range(len(self.xuanran)):
                    image_path = os.path.join(self.xuanran[i], "{}.png".format(id))

                    element.append(image_path)

                self.xuanran_lst.append(random.sample(element,1)[0])



            self.images_lst.extend(self.xuanran_lst)


        self.label_lst = []
        for id in self.items:
            image_path = os.path.join(self.gt_filepath, "{}.png".format(id))
            self.label_lst.append(image_path)


        self.label_lst.extend(self.label_lst)



        ignore_label = -1
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        self.class_16 = False
        self.class_13 = False

        print("{} num images in GTA5 {} set have been loaded.".format(len(self.label_lst), self.split))
        self.total_num = len(self.images_lst)

    def __getitem__(self, idx):



        # image_path = os.path.join(self.image_filepath, "{0:05d}.png".format(id))
        # image_path = os.path.join(self.image_filepath, "{}.png".format(id))
        image_path = self.images_lst[idx]
        image = Image.open(image_path).convert("RGB")



        # gt_image_path = os.path.join(self.gt_filepath, "{0:05d}.png".format(id))
        # gt_image_path = os.path.join(self.gt_filepath, "{}.png".format(id))

        gt_image_path = self.label_lst[idx]
        gt_image = Image.open(gt_image_path)

        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.training:
            image, gt_image = self._train_sync_transform(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform(image, gt_image)


        return image, gt_image, idx

    def __len__(self):
        return self.total_num


class GTA5_xuanran_DataLoader():
    def __init__(self, args, training=True):

        self.args = args

        data_set = GTA5_xuanran_Dataset(args,
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

        # val_split = 'val' if self.args.split == "train" else 'test'
        # val_set = GTA5_xuanran_Dataset(args,
        #                     data_root_path=args.data_root_path,
        #                     list_path=args.list_path,
        #                     split=val_split,
        #                     base_size=args.base_size,
        #                     crop_size=args.crop_size,
        #                     training=False)
        # self.val_loader = data.DataLoader(val_set,
        #                                     batch_size=self.args.batch_size,
        #                                     shuffle=False,
        #                                     num_workers=self.args.data_loader_workers,
        #                                     pin_memory=self.args.pin_memory,
        #                                     drop_last=True)
        # self.valid_iterations = (len(val_set) + self.args.batch_size) // self.args.batch_size

        self.num_iterations = (len(data_set) + self.args.batch_size) // self.args.batch_size
        # self.num_iterations = 50




class Mix_Dataset(Beiyong_Dataset):
    def __init__(self,
                 args,
                 data_root_path='../../DATASETS/datasets_seg/GTA5',
                 list_path='../datasets/GTA5/list',
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True):
        self.xuanran = args.xuanran_path

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
            raise Warning("split must be train/val/trainval/test/all")

        self.image_filepath = os.path.join(self.data_path, "images")

        self.gt_filepath = os.path.join(self.data_path, "labels")

        # self.items = [int(id.strip()) for id in open(item_list_filepath)]
        self.items = [id.strip() for id in open(item_list_filepath)]
        self.images_lst = []
        for id in self.items:
            image_path = os.path.join(self.image_filepath, "{}.png".format(id))
            self.images_lst.append(image_path)


        if self.xuanran is not None:
            self.xuanran_lst = []



            for id in self.items:
                element = []
                for i in range(len(self.xuanran)):
                    image_path = os.path.join(self.xuanran[i], "{}.png".format(id))

                    element.append(image_path)

                self.xuanran_lst.append(random.sample(element,1)[0])


            self.images_lst.extend(self.xuanran_lst)


        self.label_lst = []
        for id in self.items:
            image_path = os.path.join(self.gt_filepath, "{}.png".format(id))
            self.label_lst.append(image_path)


        self.label_lst.extend(self.label_lst)

        self.raw_lst = []
        for id in self.items:
            image_path = os.path.join(self.image_filepath, "{}.png".format(id))
            self.raw_lst.append(image_path)
        self.raw_lst.extend(self.raw_lst)




        ignore_label = -1
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        self.class_16 = False
        self.class_13 = False

        print("{} num images in GTA5 {} set have been loaded!!!!!".format(len(self.images_lst), self.split))
        self.total_num = len(self.images_lst)

    def __getitem__(self, idx):


        alpha = self.args.alpha
        # image_path = os.path.join(self.image_filepath, "{0:05d}.png".format(id))
        # image_path = os.path.join(self.image_filepath, "{}.png".format(id))
        image_path = self.images_lst[idx]
        image = Image.open(image_path).convert("RGB")

        image_raw = Image.open(self.raw_lst[idx]).convert("RGB")



        # gt_image_path = os.path.join(self.gt_filepath, "{0:05d}.png".format(id))
        # gt_image_path = os.path.join(self.gt_filepath, "{}.png".format(id))

        gt_image_path = self.label_lst[idx]
        gt_image_0 = Image.open(gt_image_path)

        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.training:
            image, gt_image = self._train_sync_transform(image,image_raw, gt_image_0)

        else:
            image, gt_image = self._val_sync_transform(image,gt_image_0)
            #image_raw, gt_image = self._train_sync_transform(image_raw, gt_image_0)
            #image = image * (1 - alpha)
            #img_raw = image_raw * alpha
            #image = torch.add(image, img_raw)

        return image, gt_image, idx

    def __len__(self):
        return self.total_num


class Mix_DataLoader():
    def __init__(self, args, training=True):

        self.args = args

        data_set = Mix_Dataset(args,
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

        # val_split = 'val' if self.args.split == "train" else 'test'
        # val_set = GTA5_xuanran_Dataset(args,
        #                     data_root_path=args.data_root_path,
        #                     list_path=args.list_path,
        #                     split=val_split,
        #                     base_size=args.base_size,
        #                     crop_size=args.crop_size,
        #                     training=False)
        # self.val_loader = data.DataLoader(val_set,
        #                                     batch_size=self.args.batch_size,
        #                                     shuffle=False,
        #                                     num_workers=self.args.data_loader_workers,
        #                                     pin_memory=self.args.pin_memory,
        #                                     drop_last=True)
        # self.valid_iterations = (len(val_set) + self.args.batch_size) // self.args.batch_size

        self.num_iterations = (len(data_set) + self.args.batch_size) // self.args.batch_size
        # self.num_iterations = 50





class Only_xuanran_Dataset(City_Dataset):
    def __init__(self,
                 args,
                 data_root_path='../datasets/GTA5',
                 list_path='../datasets/GTA5/list',
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True):
        self.xuanran = args.xuanran_path

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
            raise Warning("split must be train/val/trainval/test/all")

        self.image_filepath = os.path.join(self.data_path, "images")

        self.gt_filepath = os.path.join(self.data_path, "labels")

        # self.items = [int(id.strip()) for id in open(item_list_filepath)]
        self.items = [id.strip() for id in open(item_list_filepath)]
        self.images_lst = []
        for id in self.items:
            image_path = os.path.join(self.image_filepath, "{}.png".format(id))
            self.images_lst.append(image_path)


        if self.xuanran is not None:
            self.xuanran_lst = []



            for id in self.items:
                element = []
                for i in range(len(self.xuanran)):
                    image_path = os.path.join(self.xuanran[i], "{}.png".format(id))

                    element.append(image_path)

                self.xuanran_lst.append(random.sample(element,1)[0])

        self.images_lst = self.xuanran_lst




        self.label_lst = []
        for id in self.items:
            image_path = os.path.join(self.gt_filepath, "{}.png".format(id))
            self.label_lst.append(image_path)







        ignore_label = -1
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        self.class_16 = False
        self.class_13 = False

        print("{} num images in GTA5 {} set have been loaded.".format(len(self.label_lst), self.split))
        self.total_num = len(self.images_lst)

    def __getitem__(self, idx):



        # image_path = os.path.join(self.image_filepath, "{0:05d}.png".format(id))
        # image_path = os.path.join(self.image_filepath, "{}.png".format(id))
        image_path = self.images_lst[idx]

        image = Image.open(image_path).convert("RGB")

        # gt_image_path = os.path.join(self.gt_filepath, "{0:05d}.png".format(id))
        # gt_image_path = os.path.join(self.gt_filepath, "{}.png".format(id))

        gt_image_path = self.label_lst[idx]
        gt_image = Image.open(gt_image_path)

        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.training:
            image, gt_image = self._train_sync_transform(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform(image, gt_image)

        return image, gt_image, idx

    def __len__(self):
        return self.total_num


class Only_xuanran_DataLoader():
    def __init__(self, args, training=True):

        self.args = args

        data_set = Only_xuanran_Dataset(args,
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

        # val_split = 'val' if self.args.split == "train" else 'test'
        # val_set = GTA5_xuanran_Dataset(args,
        #                     data_root_path=args.data_root_path,
        #                     list_path=args.list_path,
        #                     split=val_split,
        #                     base_size=args.base_size,
        #                     crop_size=args.crop_size,
        #                     training=False)
        # self.val_loader = data.DataLoader(val_set,
        #                                     batch_size=self.args.batch_size,
        #                                     shuffle=False,
        #                                     num_workers=self.args.data_loader_workers,
        #                                     pin_memory=self.args.pin_memory,
        #                                     drop_last=True)
        # self.valid_iterations = (len(val_set) + self.args.batch_size) // self.args.batch_size

        self.num_iterations = (len(data_set) + self.args.batch_size) // self.args.batch_size
        # self.num_iterations = 50


class Mix_xuanran_Dataset(City_Dataset):
    def __init__(self,
                 args,
                 data_root_path='../datasets/GTA5',
                 list_path='../datasets/GTA5/list',
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True):
        self.xuanran = args.xuanran_path

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
            raise Warning("split must be train/val/trainval/test/all")

        self.image_filepath = os.path.join(self.data_path, "images")

        self.gt_filepath = os.path.join(self.data_path, "labels")

        # self.items = [int(id.strip()) for id in open(item_list_filepath)]
        self.items = [id.strip() for id in open(item_list_filepath)]
        self.images_lst = []
        for id in self.items:
            image_path = os.path.join(self.image_filepath, "{}.png".format(id))
            self.images_lst.append(image_path)


        if self.xuanran is not None:
            self.xuanran_lst = []



            for id in self.items:
                element = []
                for i in range(len(self.xuanran)):
                    image_path = os.path.join(self.xuanran[i], "{}.png".format(id))

                    element.append(image_path)

                self.xuanran_lst.append(random.sample(element,1)[0])

        self.images_lst = self.xuanran_lst




        self.label_lst = []
        for id in self.items:
            image_path = os.path.join(self.gt_filepath, "{}.png".format(id))
            self.label_lst.append(image_path)







        ignore_label = -1
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        self.class_16 = False
        self.class_13 = False

        print("{} num images in GTA5 {} set have been loaded.".format(len(self.label_lst), self.split))
        self.total_num = len(self.images_lst)

    def __getitem__(self, idx):



        # image_path = os.path.join(self.image_filepath, "{0:05d}.png".format(id))
        # image_path = os.path.join(self.image_filepath, "{}.png".format(id))
        image_path = self.images_lst[idx]

        image = Image.open(image_path).convert("RGB")

        # gt_image_path = os.path.join(self.gt_filepath, "{0:05d}.png".format(id))
        # gt_image_path = os.path.join(self.gt_filepath, "{}.png".format(id))

        gt_image_path = self.label_lst[idx]
        gt_image = Image.open(gt_image_path)

        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.training:
            image, gt_image = self._train_sync_transform(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform(image, gt_image)

        return image, gt_image, idx

    def __len__(self):
        return self.total_num


class Mix_xuanran_DataLoader():
    def __init__(self, args, training=True):

        self.args = args

        data_set = Mix_xuanran_Dataset(args,
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

        # val_split = 'val' if self.args.split == "train" else 'test'
        # val_set = GTA5_xuanran_Dataset(args,
        #                     data_root_path=args.data_root_path,
        #                     list_path=args.list_path,
        #                     split=val_split,
        #                     base_size=args.base_size,
        #                     crop_size=args.crop_size,
        #                     training=False)
        # self.val_loader = data.DataLoader(val_set,
        #                                     batch_size=self.args.batch_size,
        #                                     shuffle=False,
        #                                     num_workers=self.args.data_loader_workers,
        #                                     pin_memory=self.args.pin_memory,
        #                                     drop_last=True)
        # self.valid_iterations = (len(val_set) + self.args.batch_size) // self.args.batch_size

        self.num_iterations = (len(data_set) + self.args.batch_size) // self.args.batch_size
        # self.num_iterations = 50









class CowMix_Dataset_style_1_2(Beiyong_Dataset):
    def __init__(self,
                 args,
                 data_root_path='../datasets/GTA5',
                 list_path='../datasets/GTA5/list',
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True):
        self.xuanran = args.xuanran_path

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
            raise Warning("split must be train/val/trainval/test/all")

        self.image_filepath = os.path.join(self.data_path, "images")

        self.gt_filepath = os.path.join(self.data_path, "labels")

        # self.items = [int(id.strip()) for id in open(item_list_filepath)]
        self.items = [id.strip() for id in open(item_list_filepath)]
        self.images_lst_1 = []
        self.images_lst_2 = []
        for id in self.items:
            image_path = os.path.join(self.image_filepath, "{}.png".format(id))
            self.images_lst_1.append(image_path)

        for id in self.items:
            image_path = os.path.join(self.image_filepath, "{}.png".format(id))
            self.images_lst_2.append(image_path)

        if self.xuanran is not None:
            self.xuanran_lst_1 = []



            for id in self.items:
                element = []
                for i in range(len(self.xuanran)):
                    image_path = os.path.join(self.xuanran[i], "{}.png".format(id))

                    element.append(image_path)

                self.xuanran_lst_1.append(random.sample(element,1)[0])



            self.images_lst_1.extend(self.xuanran_lst_1)

        if self.xuanran is not None:
            self.xuanran_lst_2 = []



            for id in self.items:
                element = []
                for i in range(len(self.xuanran)):
                    image_path = os.path.join(self.xuanran[i], "{}.png".format(id))

                    element.append(image_path)

                self.xuanran_lst_2.append(random.sample(element,1)[0])



            self.images_lst_2.extend(self.xuanran_lst_2)


        self.label_lst = []
        for id in self.items:
            image_path = os.path.join(self.gt_filepath, "{}.png".format(id))
            self.label_lst.append(image_path)


        self.label_lst.extend(self.label_lst)

        self.raw_lst = []
        for id in self.items:
            image_path = os.path.join(self.image_filepath, "{}.png".format(id))
            self.raw_lst.append(image_path)
        self.raw_lst.extend(self.raw_lst)




        ignore_label = -1
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        self.class_16 = False
        self.class_13 = False

        print("{} num images in GTA5 {} set have been loaded.".format(len(self.label_lst), self.split))
        self.total_num = len(self.images_lst_1)

    def __getitem__(self, idx):


        alpha = self.args.alpha
        # image_path = os.path.join(self.image_filepath, "{0:05d}.png".format(id))
        # image_path = os.path.join(self.image_filepath, "{}.png".format(id))
        image_path = self.images_lst_1[idx]
        image_1 = Image.open(image_path).convert("RGB")

        image_path = self.images_lst_2[idx]
        image_2 = Image.open(image_path).convert("RGB")

        image_raw = Image.open(self.raw_lst[idx]).convert("RGB")



        # gt_image_path = os.path.join(self.gt_filepath, "{0:05d}.png".format(id))
        # gt_image_path = os.path.join(self.gt_filepath, "{}.png".format(id))

        gt_image_path = self.label_lst[idx]
        gt_image_0 = Image.open(gt_image_path)

        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.training:
            image, gt_image = self._train_s1_s2_transform(image_1,image_2,image_raw, gt_image_0)

        else:
            image, gt_image = self._val_sync_transform(image_1,gt_image_0)


        return image, gt_image, idx

    def __len__(self):
        return self.total_num


class CowMix_s1_s2_DataLoader():
    def __init__(self, args, training=True):

        self.args = args

        data_set = CowMix_Dataset_style_1_2(args,
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



        self.num_iterations = (len(data_set) + self.args.batch_size) // self.args.batch_size


class CowMix_Dataset_r_s(Beiyong_Dataset):
    def __init__(self,
                 args,
                 data_root_path='../datasets/GTA5',
                 list_path='../datasets/GTA5/list',
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True):
        self.xuanran = args.xuanran_path

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
            raise Warning("split must be train/val/trainval/test/all")

        self.image_filepath = os.path.join(self.data_path, "images")

        self.gt_filepath = os.path.join(self.data_path, "labels")

        # self.items = [int(id.strip()) for id in open(item_list_filepath)]
        self.items = [id.strip() for id in open(item_list_filepath)]
        self.images_lst_1 = []

        for id in self.items:
            image_path = os.path.join(self.image_filepath, "{}.png".format(id))
            self.images_lst_1.append(image_path)



        if self.xuanran is not None:
            self.xuanran_lst_1 = []



            for id in self.items:
                element = []
                for i in range(len(self.xuanran)):
                    image_path = os.path.join(self.xuanran[i], "{}.png".format(id))

                    element.append(image_path)

                self.xuanran_lst_1.append(random.sample(element,1)[0])



            self.images_lst_1.extend(self.xuanran_lst_1)



        self.label_lst = []
        for id in self.items:
            image_path = os.path.join(self.gt_filepath, "{}.png".format(id))
            self.label_lst.append(image_path)


        self.label_lst.extend(self.label_lst)

        self.raw_lst = []
        for id in self.items:
            image_path = os.path.join(self.image_filepath, "{}.png".format(id))
            self.raw_lst.append(image_path)
        self.raw_lst.extend(self.raw_lst)




        ignore_label = -1
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        self.class_16 = False
        self.class_13 = False

        print("{} num images in GTA5 {} set have been loaded.".format(len(self.label_lst), self.split))
        self.total_num = len(self.images_lst_1)

    def __getitem__(self, idx):


        alpha = self.args.alpha
        # image_path = os.path.join(self.image_filepath, "{0:05d}.png".format(id))
        # image_path = os.path.join(self.image_filepath, "{}.png".format(id))
        image_path = self.images_lst_1[idx]
        image_1 = Image.open(image_path).convert("RGB")



        image_raw = Image.open(self.raw_lst[idx]).convert("RGB")



        # gt_image_path = os.path.join(self.gt_filepath, "{0:05d}.png".format(id))
        # gt_image_path = os.path.join(self.gt_filepath, "{}.png".format(id))

        gt_image_path = self.label_lst[idx]
        gt_image_0 = Image.open(gt_image_path)

        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.training:
            image, gt_image = self._train_r_s_transform(image_1,image_raw, gt_image_0)

        else:
            image, gt_image = self._val_sync_transform(image_1,gt_image_0)


        return image, gt_image, idx

    def __len__(self):
        return self.total_num


class CowMix_r_s_DataLoader():
    def __init__(self, args, training=True):

        self.args = args

        data_set = CowMix_Dataset_r_s(args,
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



        self.num_iterations = (len(data_set) + self.args.batch_size) // self.args.batch_size




class PS_Dataset(Beiyong_Dataset):
    def __init__(self,
                 args,
                 data_root_path='../datasets/GTA5',
                 list_path='../datasets/GTA5/list',
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True):
        self.xuanran = args.xuanran_path

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
            raise Warning("split must be train/val/trainval/test/all")

        self.image_filepath = os.path.join(self.data_path, "images")

        self.gt_filepath = os.path.join(self.data_path, "labels")

        # self.items = [int(id.strip()) for id in open(item_list_filepath)]
        self.items = [id.strip() for id in open(item_list_filepath)]
        self.images_lst = []
        for id in self.items:
            image_path = os.path.join(self.image_filepath, "{}.png".format(id))
            self.images_lst.append(image_path)


        if self.xuanran is not None:
            self.xuanran_lst = []



            for id in self.items:
                element = []
                for i in range(len(self.xuanran)):
                    image_path = os.path.join(self.xuanran[i], "{}.png".format(id))

                    element.append(image_path)

                self.xuanran_lst.append(random.sample(element,1)[0])



            self.images_lst.extend(self.xuanran_lst)


        self.label_lst = []
        for id in self.items:
            image_path = os.path.join(self.gt_filepath, "{}.png".format(id))
            self.label_lst.append(image_path)


        self.label_lst.extend(self.label_lst)

        self.raw_lst = []
        for id in self.items:
            image_path = os.path.join(self.image_filepath, "{}.png".format(id))
            self.raw_lst.append(image_path)
        self.raw_lst.extend(self.raw_lst)




        ignore_label = -1
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        self.class_16 = False
        self.class_13 = False

        print("{} num images in GTA5 {} set have been loaded.".format(len(self.label_lst), self.split))
        self.total_num = len(self.images_lst)

    def __getitem__(self, idx):


        alpha = self.args.alpha
        # image_path = os.path.join(self.image_filepath, "{0:05d}.png".format(id))
        # image_path = os.path.join(self.image_filepath, "{}.png".format(id))
        image_path = self.images_lst[idx]
        image = Image.open(image_path).convert("RGB")

        image_raw = Image.open(self.raw_lst[idx]).convert("RGB")



        # gt_image_path = os.path.join(self.gt_filepath, "{0:05d}.png".format(id))
        # gt_image_path = os.path.join(self.gt_filepath, "{}.png".format(id))

        gt_image_path = self.label_lst[idx]
        gt_image_0 = Image.open(gt_image_path)

        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.training:
            image, image_raw, gt_image = self._train_PS_transform(image,image_raw, gt_image_0)

        else:
            image, gt_image = self._val_sync_transform(image,gt_image_0)
            #image_raw, gt_image = self._train_sync_transform(image_raw, gt_image_0)
            #image = image * (1 - alpha)
            #img_raw = image_raw * alpha
            #image = torch.add(image, img_raw)

        return image,  image_raw, gt_image, idx

    def __len__(self):
        return self.total_num


class PartStyle_DataLoader():
    def __init__(self, args, training=True):

        self.args = args

        data_set = PS_Dataset(args,
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

        # val_split = 'val' if self.args.split == "train" else 'test'
        # val_set = GTA5_xuanran_Dataset(args,
        #                     data_root_path=args.data_root_path,
        #                     list_path=args.list_path,
        #                     split=val_split,
        #                     base_size=args.base_size,
        #                     crop_size=args.crop_size,
        #                     training=False)
        # self.val_loader = data.DataLoader(val_set,
        #                                     batch_size=self.args.batch_size,
        #                                     shuffle=False,
        #                                     num_workers=self.args.data_loader_workers,
        #                                     pin_memory=self.args.pin_memory,
        #                                     drop_last=True)
        # self.valid_iterations = (len(val_set) + self.args.batch_size) // self.args.batch_size

        self.num_iterations = (len(data_set) + self.args.batch_size) // self.args.batch_size
        # self.num_iterations = 50






class Raw_TR_Dataset(Beiyong_Dataset):
    def __init__(self,
                 args,
                 data_root_path='../datasets/GTA5',
                 list_path='../datasets/GTA5/list',
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True):
        self.xuanran = args.xuanran_path

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
            raise Warning("split must be train/val/trainval/test/all")

        self.image_filepath = os.path.join(self.data_path, "images")

        self.gt_filepath = os.path.join(self.data_path, "labels")

        # self.items = [int(id.strip()) for id in open(item_list_filepath)]
        self.items = [id.strip() for id in open(item_list_filepath)]
        self.images_lst = []
        # for id in self.items:
        #     image_path = os.path.join(self.image_filepath, "{}.png".format(id))
        #     self.images_lst.append(image_path)


        if self.xuanran is not None:
            self.xuanran_lst = []



            for id in self.items:
                element = []
                for i in range(len(self.xuanran)):
                    image_path = os.path.join(self.xuanran[i], "{}.png".format(id))

                    element.append(image_path)

                self.xuanran_lst.append(random.sample(element,1)[0])



            self.images_lst.extend(self.xuanran_lst)


        self.label_lst = []
        for id in self.items:
            image_path = os.path.join(self.gt_filepath, "{}.png".format(id))
            self.label_lst.append(image_path)


        # self.label_lst.extend(self.label_lst)

        self.raw_lst = []
        for id in self.items:
            image_path = os.path.join(self.image_filepath, "{}.png".format(id))
            self.raw_lst.append(image_path)
        # self.raw_lst.extend(self.raw_lst)




        ignore_label = -1
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        self.class_16 = False
        self.class_13 = False

        print("{} num images in GTA5 {} set have been loaded.".format(len(self.label_lst), self.split))
        self.total_num = len(self.images_lst)

    def __getitem__(self, idx):


        alpha = self.args.alpha
        # image_path = os.path.join(self.image_filepath, "{0:05d}.png".format(id))
        # image_path = os.path.join(self.image_filepath, "{}.png".format(id))
        image_path = self.images_lst[idx]
        image = Image.open(image_path).convert("RGB")

        image_raw = Image.open(self.raw_lst[idx]).convert("RGB")



        # gt_image_path = os.path.join(self.gt_filepath, "{0:05d}.png".format(id))
        # gt_image_path = os.path.join(self.gt_filepath, "{}.png".format(id))

        gt_image_path = self.label_lst[idx]
        gt_image_0 = Image.open(gt_image_path)

        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.training:
            image, image_raw, gt_image = self._train_PS_transform(image,image_raw, gt_image_0)

        else:
            image, gt_image = self._val_sync_transform(image,gt_image_0)
            #image_raw, gt_image = self._train_sync_transform(image_raw, gt_image_0)
            #image = image * (1 - alpha)
            #img_raw = image_raw * alpha
            #image = torch.add(image, img_raw)

        return image,  image_raw, gt_image, idx

    def __len__(self):
        return self.total_num


class Raw_TR_DataLoader():
    def __init__(self, args, training=True):

        self.args = args

        data_set = Raw_TR_Dataset(args,
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

        # val_split = 'val' if self.args.split == "train" else 'test'
        # val_set = GTA5_xuanran_Dataset(args,
        #                     data_root_path=args.data_root_path,
        #                     list_path=args.list_path,
        #                     split=val_split,
        #                     base_size=args.base_size,
        #                     crop_size=args.crop_size,
        #                     training=False)
        # self.val_loader = data.DataLoader(val_set,
        #                                     batch_size=self.args.batch_size,
        #                                     shuffle=False,
        #                                     num_workers=self.args.data_loader_workers,
        #                                     pin_memory=self.args.pin_memory,
        #                                     drop_last=True)
        # self.valid_iterations = (len(val_set) + self.args.batch_size) // self.args.batch_size

        self.num_iterations = (len(data_set) + self.args.batch_size) // self.args.batch_size
        # self.num_iterations = 50




class R_SS_Dataset(Beiyong_Dataset):
    def __init__(self,
                 args,
                 data_root_path='../datasets/GTA5',
                 list_path='../datasets/GTA5/list',
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True):
        self.xuanran = args.xuanran_path

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
            raise Warning("split must be train/val/trainval/test/all")

        self.image_filepath = os.path.join(self.data_path, "images")

        self.gt_filepath = os.path.join(self.data_path, "labels")

        # self.items = [int(id.strip()) for id in open(item_list_filepath)]
        self.items = [id.strip() for id in open(item_list_filepath)]
        self.images_lst = []
        self.images_lst_2 = []
        for id in self.items:
            image_path = os.path.join(self.image_filepath, "{}.png".format(id))
            self.images_lst.append(image_path)
            self.images_lst_2.append(image_path)


        if self.xuanran is not None:
            self.xuanran_lst = []
            self.xuanran_lst_2 = []



            for id in self.items:
                element = []
                for i in range(len(self.xuanran)):
                    image_path = os.path.join(self.xuanran[i], "{}.png".format(id))

                    element.append(image_path)

                self.xuanran_lst.append(random.sample(element,1)[0])
                self.xuanran_lst_2.append(random.sample(element,1)[0])



            self.images_lst.extend(self.xuanran_lst)
            self.images_lst_2.extend(self.xuanran_lst_2)


        self.label_lst = []
        for id in self.items:
            image_path = os.path.join(self.gt_filepath, "{}.png".format(id))
            self.label_lst.append(image_path)


        self.label_lst.extend(self.label_lst)

        self.raw_lst = []
        for id in self.items:
            image_path = os.path.join(self.image_filepath, "{}.png".format(id))
            self.raw_lst.append(image_path)
        self.raw_lst.extend(self.raw_lst)




        ignore_label = -1
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        self.class_16 = False
        self.class_13 = False

        print("{} num images in GTA5 {} set have been loaded.".format(len(self.label_lst), self.split))
        self.total_num = len(self.images_lst)

    def __getitem__(self, idx):


        alpha = self.args.alpha
        # image_path = os.path.join(self.image_filepath, "{0:05d}.png".format(id))
        # image_path = os.path.join(self.image_filepath, "{}.png".format(id))
        image_path = self.images_lst[idx]
        image = Image.open(image_path).convert("RGB")
        image_path_2 = self.images_lst_2[idx]
        image_2 = Image.open(image_path_2).convert("RGB")


        image_raw = Image.open(self.raw_lst[idx]).convert("RGB")



        # gt_image_path = os.path.join(self.gt_filepath, "{0:05d}.png".format(id))
        # gt_image_path = os.path.join(self.gt_filepath, "{}.png".format(id))

        gt_image_path = self.label_lst[idx]
        gt_image_0 = Image.open(gt_image_path)

        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.training:
            image, image_2, image_raw, gt_image = self._train_RSS_transform(image,image_2,image_raw, gt_image_0)

        else:
            image, gt_image = self._val_sync_transform(image,gt_image_0)
            #image_raw, gt_image = self._train_sync_transform(image_raw, gt_image_0)
            #image = image * (1 - alpha)
            #img_raw = image_raw * alpha
            #image = torch.add(image, img_raw)

        return image,image_2, image_raw, gt_image, idx

    def __len__(self):
        return self.total_num


class RSS_DataLoader():
    def __init__(self, args, training=True):

        self.args = args

        data_set = R_SS_Dataset(args,
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

        # val_split = 'val' if self.args.split == "train" else 'test'
        # val_set = GTA5_xuanran_Dataset(args,
        #                     data_root_path=args.data_root_path,
        #                     list_path=args.list_path,
        #                     split=val_split,
        #                     base_size=args.base_size,
        #                     crop_size=args.crop_size,
        #                     training=False)
        # self.val_loader = data.DataLoader(val_set,
        #                                     batch_size=self.args.batch_size,
        #                                     shuffle=False,
        #                                     num_workers=self.args.data_loader_workers,
        #                                     pin_memory=self.args.pin_memory,
        #                                     drop_last=True)
        # self.valid_iterations = (len(val_set) + self.args.batch_size) // self.args.batch_size

        self.num_iterations = (len(data_set) + self.args.batch_size) // self.args.batch_size
        # self.num_iterations = 50



class F_R_SS_Dataset(Beiyong_Dataset):
    def __init__(self,
                 args,
                 data_root_path='../datasets/GTA5',
                 list_path='../datasets/GTA5/list',
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True):
        self.xuanran = args.xuanran_path

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
            raise Warning("split must be train/val/trainval/test/all")

        self.image_filepath = os.path.join(self.data_path, "images")

        self.gt_filepath = os.path.join(self.data_path, "labels")

        # self.items = [int(id.strip()) for id in open(item_list_filepath)]
        self.items = [id.strip() for id in open(item_list_filepath)]
        self.images_lst = []
        self.images_lst_2 = []
        for id in self.items:
            image_path = os.path.join(self.image_filepath, "{}.png".format(id))
            self.images_lst.append(image_path)
            self.images_lst_2.append(image_path)


        if self.xuanran is not None:
            self.xuanran_lst = []
            self.xuanran_lst_2 = []



            for id in self.items:
                element = []
                for i in range(len(self.xuanran)):
                    image_path = os.path.join(self.xuanran[i], "{}.png".format(id))

                    element.append(image_path)

                self.xuanran_lst.append(random.sample(element,1)[0])
                self.xuanran_lst_2.append(random.sample(element,1)[0])



            self.images_lst.extend(self.xuanran_lst)
            self.images_lst_2.extend(self.xuanran_lst_2)


        self.label_lst = []
        for id in self.items:
            image_path = os.path.join(self.gt_filepath, "{}.png".format(id))
            self.label_lst.append(image_path)


        self.label_lst.extend(self.label_lst)

        self.raw_lst = []
        for id in self.items:
            image_path = os.path.join(self.image_filepath, "{}.png".format(id))
            self.raw_lst.append(image_path)
        self.raw_lst.extend(self.raw_lst)




        ignore_label = -1
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        self.class_16 = False
        self.class_13 = False

        print("{} num images in GTA5 {} set have been loaded.".format(len(self.label_lst), self.split))
        self.total_num = len(self.images_lst)

    def __getitem__(self, idx):


        alpha = self.args.alpha
        # image_path = os.path.join(self.image_filepath, "{0:05d}.png".format(id))
        # image_path = os.path.join(self.image_filepath, "{}.png".format(id))
        image_path = self.images_lst[idx]
        image = Image.open(image_path).convert("RGB")
        image_path_2 = self.images_lst_2[idx]
        image_2 = Image.open(image_path_2).convert("RGB")


        image_raw = Image.open(self.raw_lst[idx]).convert("RGB")



        # gt_image_path = os.path.join(self.gt_filepath, "{0:05d}.png".format(id))
        # gt_image_path = os.path.join(self.gt_filepath, "{}.png".format(id))

        gt_image_path = self.label_lst[idx]
        gt_image_0 = Image.open(gt_image_path)

        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.training:
            image, image_2, image_raw, gt_image = self._Fa_RSS_transform(image,image_2,image_raw, gt_image_0)

        else:
            image, gt_image = self._val_sync_transform(image,gt_image_0)
            #image_raw, gt_image = self._train_sync_transform(image_raw, gt_image_0)
            #image = image * (1 - alpha)
            #img_raw = image_raw * alpha
            #image = torch.add(image, img_raw)

        return image,image_2, image_raw, gt_image, idx

    def __len__(self):
        return self.total_num


class F_RSS_DataLoader():
    def __init__(self, args, training=True):

        self.args = args

        data_set = F_R_SS_Dataset(args,
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

        # val_split = 'val' if self.args.split == "train" else 'test'
        # val_set = GTA5_xuanran_Dataset(args,
        #                     data_root_path=args.data_root_path,
        #                     list_path=args.list_path,
        #                     split=val_split,
        #                     base_size=args.base_size,
        #                     crop_size=args.crop_size,
        #                     training=False)
        # self.val_loader = data.DataLoader(val_set,
        #                                     batch_size=self.args.batch_size,
        #                                     shuffle=False,
        #                                     num_workers=self.args.data_loader_workers,
        #                                     pin_memory=self.args.pin_memory,
        #                                     drop_last=True)
        # self.valid_iterations = (len(val_set) + self.args.batch_size) // self.args.batch_size

        self.num_iterations = (len(data_set) + self.args.batch_size) // self.args.batch_size
        # self.num_iterations = 50



class Print_Dataset(Beiyong_Dataset):
    def __init__(self,
                 args,
                 data_root_path='../datasets/GTA5',
                 list_path='../datasets/GTA5/list',
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True):
        self.xuanran = args.xuanran_path

        self.args = args
        self.data_path=data_root_path
        self.list_path=list_path
        self.split=split
        self.base_size=base_size
        self.crop_size=crop_size

        self.base_size = self.base_size if isinstance(self.base_size, tuple) else (self.base_size, self.base_size)
        self.crop_size = self.crop_size if isinstance(self.crop_size, tuple) else (self.crop_size, self.crop_size)
        self.training = training


        self.random_crop = args.random_crop
        self.resize = args.resize


        item_list_filepath = os.path.join(self.list_path, self.split+".txt")

        if not os.path.exists(item_list_filepath):
            raise Warning("split must be train/val/trainval/test/all")

        self.image_filepath = os.path.join(self.data_path, "images")
        self.gt_filepath = os.path.join(self.data_path, "labels")


        # self.items = [int(id.strip()) for id in open(item_list_filepath)]
        self.items = [id.strip() for id in open(item_list_filepath)]
        self.images_lst = []
        self.images_lst_2 = []
        for id in self.items:
            image_path = os.path.join(self.image_filepath, "{}.png".format(id))
            self.images_lst.append(image_path)



        if self.xuanran is not None:
            self.xuanran_lst = []




            for id in self.items:
                element = []
                for i in range(len(self.xuanran)):
                    image_path = os.path.join(self.xuanran[i], "{}.png".format(id))

                    element.append(image_path)

                self.xuanran_lst.append(random.sample(element,1)[0])








        self.raw_lst = []
        for id in self.items:
            image_path = os.path.join(self.image_filepath, "{}.png".format(id))
            self.raw_lst.append(image_path)


        self.label_lst = []
        for id in self.items:
            image_path = os.path.join(self.gt_filepath, "{}.png".format(id))
            self.label_lst.append(image_path)


        ignore_label = -1
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        self.class_16 = False
        self.class_13 = False


        self.total_num = len(self.images_lst)

    def __getitem__(self, idx):

        loader = ttransforms.Compose([ttransforms.ToTensor()])
        alpha = self.args.alpha
        # image_path = os.path.join(self.image_filepath, "{0:05d}.png".format(id))
        # image_path = os.path.join(self.image_filepath, "{}.png".format(id))
        image_path = self.xuanran_lst[idx]
        image = Image.open(image_path).convert("RGB")
        image_raw = Image.open(self.raw_lst[idx]).convert("RGB")
        gt_image_path = self.label_lst[idx]
        gt_image = Image.open(gt_image_path).convert("RGB")
        image = loader(image)
        image_raw = loader(image_raw)
        gt_image = loader(gt_image)

        image = image * (1 - self.args.alpha)
        img_raw_w = image_raw * self.args.alpha
        img = torch.add(image, img_raw_w)





        return img, image_raw, gt_image, idx

    def __len__(self):
        return self.total_num


class Print_DataLoader():
    def __init__(self, args, training=True):

        self.args = args

        data_set = Print_Dataset(args,
                                data_root_path=args.data_root_path,
                                list_path=args.list_path,
                                split=args.split,
                                base_size=args.base_size,
                                crop_size=args.crop_size,
                                training=training)


        if self.args.split == "train" or self.args.split == "trainval" or self.args.split =="all":
            self.data_loader = data.DataLoader(data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=False,
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


        self.num_iterations = (len(data_set) + self.args.batch_size) // self.args.batch_size
        # self.num_iterations = 50

