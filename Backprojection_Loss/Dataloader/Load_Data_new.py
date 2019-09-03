"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import torch
from torch.autograd import Variable
from PIL import Image, ImageOps
import cv2
import json
import numbers
import random
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.utils.data.dataloader import default_collate
warnings.simplefilter('ignore', np.RankWarning)
from torch.utils.data.sampler import Sampler




def get_testloader(path, batch_size, num_workers, resize=256):
    json_file = os.path.join(path, 'test_label.json')
    transformed_dataset = LaneTestSet(gt_file=json_file,
                                      path=path,
                                      resize=resize)

    data_loader = DataLoader(dataset=transformed_dataset, 
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=num_workers,
                                          drop_last=False)
    return data_loader


class LaneTestSet(Dataset):
    '''Dataset used as testset'''
    def __init__(self, gt_file, resize, path):
        self.img_info = [json.loads(line) for line in open(gt_file,'r')]
        print('size test loader: ', len(self.img_info))
        self.path = path
        self.resize = resize
        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, idx):
        im_path = self.img_info[idx]['raw_file']
        img_name = os.path.join(self.path, im_path)
        with open(img_name, 'rb') as f:
            image = Image.open(f).convert('RGB')

        # Crop and resize images
        w, h = image.size
        image = F.crop(image, h-640, 0, 640, w)
        image = F.resize(image, size=(self.resize, 2*self.resize), interpolation=Image.BILINEAR)
        image = self.totensor(image).float()
        return image


class LaneDataset(Dataset):
    """Dataset with labeled lanes"""
    def __init__(self, end_to_end, valid_idx, json_file, lanes_file, image_dir, gt_dir, flip_on, resize, nclasses):
        """
        Args: valid_idx (list)  : Indices of validation images
              json_file (string): File with labels
              image_dir (string): Path to the images
              gt_dir    (string): Path to the ground truth images
              flip on   (bool)  : Boolean to flip images randomly
              end_to_end (bool) : Boolean to compute lane line params end to end
              resize    (int)   : Height of resized image
        """
        line_file = 'Labels/label_new.json'
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.valid_idx = valid_idx
        self.flip_on = flip_on
        self.resize = resize
        self.totensor = transforms.ToTensor()
        self.params = [json.loads(line) for line in open(json_file).readlines()]
        self.ordered_lanes = [json.loads(line) for line in open(lanes_file).readlines()]
        self.line_file = [json.loads(line) for line in open(line_file).readlines()]
        self.end_to_end = end_to_end
        self.rgb_lst = sorted(os.listdir(image_dir))
        self.gt_lst = sorted(os.listdir(gt_dir))
        self.num_imgs = len(self.rgb_lst)
        assert len(self.rgb_lst) == len(self.gt_lst) == 3626

        target_idx = [int(i.split('.')[0]) for i in self.rgb_lst]
        self.valid_idx = [target_idx[i]-1 for i in valid_idx]
        self.num_points = 56
        self.nclasses = nclasses
        print('Flipping images randomly:', self.flip_on)
        print('End to end lane detection:', self.end_to_end)

    def __len__(self):
        """
        Conventional len method
        """
        return self.num_img

    def __getitem__(self, idx):
        """
        Args: idx (int): Index in list to load image
        """
        # Load img, gt, x_coordinates, y_coordinates, line_type
        assert self.rgb_lst[idx].split('.')[0] == self.gt_lst[idx].split('.')[0]
        img_name = os.path.join(self.image_dir, self.rgb_lst[idx])
        gt_name = os.path.join(self.gt_dir, self.gt_lst[idx])
        with open(img_name, 'rb') as f:
            image = (Image.open(f).convert('RGB'))
        with open(gt_name, 'rb') as f:
            gt = (Image.open(f).convert('P'))
        idx = int(self.rgb_lst[idx].split('.')[0]) - 1
        lanes_lst = self.ordered_lanes[idx]["lanes"]
        h_samples = self.ordered_lanes[idx]["h_samples"]
        line_lst = self.line_file[idx]["lines"]

        # Crop and resize images
        w, h = image.size
        image, gt = F.crop(image, h-640, 0, 640, w), F.crop(gt, h-640, 0, 640, w)
        image = F.resize(image, size=(self.resize, 2*self.resize), interpolation=Image.BILINEAR)
        gt = F.resize(gt, size=(self.resize, 2*self.resize), interpolation=Image.NEAREST)

        # Adjust size of lanes matrix
        # lanes = np.array(lanes_lst)[:, -self.num_points:]
        lanes = np.array(lanes_lst)
        to_add = np.full((4, 56 - lanes.shape[1]), -2)
        lanes = np.hstack((to_add, lanes))

        # Get valid coordinates from lanes matrix
        valid_points = np.int32(lanes>0)
        valid_points[:, :8] = 0 # start from h-samples = 210

        # Resize coordinates
        lanes = lanes/2.5
        track = lanes < 0
        h_samples = np.array(h_samples)/2.5 - 32
        lanes[track] = -2

        # Compute horizon for resized img
        horizon_lanes = []
        for lane in lanes:
            horizon_lanes.append(min([y_cord for (x_cord, y_cord) in zip(lane, h_samples) if x_cord != -2] or [self.resize]))
        y_val = min(horizon_lanes)
        horizon = torch.zeros(gt.size[1])
        horizon[0:int(np.floor(y_val))] = 1

        # Compute line type in image
        # line_lst = np.prod(lanes == -2, axis=1) # 1 when line is not present

        gt = np.array(gt)
        idx3 = np.isin(gt, 3)
        idx4 = np.isin(gt, 4)
        if self.nclasses < 3:
            gt[idx3] = 0
            gt[idx4] = 0
        # Flip ground truth ramdomly
        hflip_input = np.random.uniform(0.0, 1.0) > 0.5 and self.flip_on
        if idx not in self.valid_idx and hflip_input:
            image, gt = F.hflip(image), np.flip(gt, axis=1)
            idx1 = np.isin(gt, 1)
            idx2 = np.isin(gt, 2)
            gt[idx1] = 2
            gt[idx2] = 1
            gt[idx3] = 4
            gt[idx4] = 3
            lanes = (2*self.resize - 1) - lanes
            lanes[track] = -2
            lanes = lanes[[1, 0, 3, 2]]
            # line_lst = np.prod(lanes == -2, axis=1)
            line_lst = mirror_list(line_lst)

        # Get Tensors
        gt = Image.fromarray(gt)
        image, gt = self.totensor(image).float(), (self.totensor(gt)*255).long()

        # Cast to correct types
        line_lst = np.array(line_lst[3:7])
        line_lst = torch.from_numpy(np.array(line_lst + 1)).clamp(0, 1).float()
        valid_points = torch.from_numpy(valid_points).double()

        lanes = torch.from_numpy(lanes).double()
        horizon = horizon.float()

        if idx in self.valid_idx:
            index = self.valid_idx.index(idx)
            return image, gt, lanes, idx, line_lst, horizon, index, valid_points
        return image, gt, lanes, idx, line_lst, horizon, valid_points


def mirror_list(lst):
    '''
    Mirror lists of lane and line classification ground truth in order to make flipping possible
    '''
    middle = len(lst)//2
    first = list(reversed(lst[:middle]))
    second = list(reversed(lst[middle:]))
    return second + first

def homogenous_transformation(Matrix, x, y):
    """
    Helper function to transform coordionates defined by transformation matrix

    Args:
            Matrix (multi dim - array): Transformation matrix
            x (array): original x coordinates
            y (array): original y coordinates
    """
    ones = np.ones((1,len(y)))
    coordinates = np.vstack((x, y, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0,:]/trans[2,:]
    y_vals = trans[1,:]/trans[2,:]
    return x_vals, y_vals


def get_homography(resize=320):
    factor = resize/640
    y_start = 0.3*(resize-1)
    y_stop = (resize-1)
    src = np.float32([[0.45*(2*resize-1),y_start],
                      [0.55*(2*resize-1), y_start],
                      [0.1*(2*resize-1),y_stop],
                      [0.9*(2*resize-1), y_stop]])
    dst = np.float32([[0.45*(2*resize-1), y_start],
                      [0.55*(2*resize-1), y_start],
                      [0.45*(2*resize-1), y_stop],
                      [0.55*(2*resize-1),y_stop]])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    return M_inv



class SequentialIndicesSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def get_loader(num_train, json_file, lanes_file, image_dir, gt_dir, flip_on, batch_size, val_batch_size,
               shuffle, num_workers, end_to_end, resize, nclasses, split_percentage=0.2):
    '''
    Splits dataset in training and validation set and creates dataloaders
    '''
    indices = list(range(num_train))
    split = int(np.floor(split_percentage*num_train))

    if shuffle is True:
        np.random.seed(num_train)
        np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    print('size train loader is', len(train_idx))
    print('size valid loader is', len(valid_idx))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = SequentialIndicesSampler(valid_idx)

    transformed_dataset = LaneDataset(end_to_end=end_to_end,
                                      valid_idx=valid_idx,
                                      json_file=json_file,
                                      lanes_file=lanes_file,
                                      image_dir=image_dir,
                                      gt_dir=gt_dir,
                                      flip_on=flip_on,
                                      resize=resize,
                                      nclasses=nclasses)

    train_loader = DataLoader(transformed_dataset,
                              batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True) #, collate_fn=my_collate)

    valid_loader = DataLoader(transformed_dataset,
                              batch_size=val_batch_size, sampler=valid_sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True) #collate_fn=my_collate)

    return train_loader, valid_loader, valid_idx


def my_collate(batch):
    batch = [dict for dict in batch if dict['4_lanes'] is True]
    return default_collate(batch)


def load_valid_set_file(valid_idx):
    file1 = "Labels/label_data_0313.json"
    file2 = "Labels/label_data_0531.json"
    file3 = "Labels/label_data_0601.json"
    labels_file1 = [json.loads(line) for line in open(file1).readlines()]
    labels_file2 = [json.loads(line) for line in open(file2).readlines()]
    labels_file3 = [json.loads(line) for line in open(file3).readlines()]
    len_file1 = len(labels_file1)
    len_file2 = len(labels_file2)

    with open('validation_set.json', 'w') as jsonFile:
        for image_id in valid_idx:
            if image_id < len_file1:
                labels = labels_file1[image_id]
            elif image_id < (len_file1 + len_file2):
                image_id_new = image_id - len_file1
                labels = labels_file2[image_id_new]
            else:
                image_id_new = image_id - len_file1 - len_file2
                labels = labels_file3[image_id_new]

            json.dump(labels, jsonFile)
            jsonFile.write('\n')


def load_valid_set_file_all(valid_idx, target_file, image_dir):
    # file1 = "Labels/Curve_parameters.json"
    file1 = "Labels/label_data_all.json"
    labels_file = [json.loads(line) for line in open(file1).readlines()]
    content = sorted(os.listdir(image_dir))
    target_idx = [int(i.split('.')[0]) for i in content]
    new_idx = [target_idx[i]-1 for i in valid_idx]
    with open(target_file, 'w') as jsonFile:
        for image_id in new_idx:
            labels = labels_file[image_id]
            json.dump(labels, jsonFile)
            jsonFile.write('\n')


def load_0313_valid_set_file(valid_idx, nclasses):
    file1 = "Labels/label_data_0313.json"
    labels_file1 = [json.loads(line) for line in open(file1).readlines()]
    with open('validation_set.json', 'w') as jsonFile:
        for image_id in valid_idx:
            labels = labels_file1[image_id]
            labels['lanes'] = labels['lanes'][0:nclasses]
            json.dump(labels, jsonFile)
            jsonFile.write('\n')


