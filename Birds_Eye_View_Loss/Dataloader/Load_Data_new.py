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


class LaneDataset(Dataset):
    """Dataset with labeled lanes"""
    def __init__(self, end_to_end, valid_idx, json_file, image_dir, gt_dir, flip_on, resize):
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
        self.line_file = [json.loads(line) for line in open(line_file).readlines()]
        self.end_to_end = end_to_end
        self.rgb_lst = sorted(os.listdir(image_dir))
        self.gt_lst = sorted(os.listdir(gt_dir))
        self.num_imgs = len(self.rgb_lst)
        assert len(self.rgb_lst) == len(self.gt_lst) == 2535

        target_idx = [int(i.split('.')[0]) for i in self.rgb_lst]
        self.valid_idx = [target_idx[i]-1 for i in valid_idx]

    def __len__(self):
        """
        Conventional len method
        """
        return self.num_img

    def __getitem__(self, idx):
        """
        Args: idx (int): Index in list to load image
        """
        assert self.rgb_lst[idx].split('.')[0] == self.gt_lst[idx].split('.')[0]
        img_name = os.path.join(self.image_dir, self.rgb_lst[idx])
        gt_name = os.path.join(self.gt_dir, self.gt_lst[idx])
        with open(img_name, 'rb') as f:
            image = (Image.open(f).convert('RGB'))
        with open(gt_name, 'rb') as f:
            gt = (Image.open(f).convert('P'))
        idx = int(self.rgb_lst[idx].split('.')[0]) - 1
        params = self.params[idx]["poly_params"]
        line_lst = self.line_file[idx]["lines"]

        w, h = image.size
        image, gt = F.crop(image, h-640, 0, 640, w), F.crop(gt, h-640, 0, 640, w)
        image = F.resize(image, size=(self.resize, 2*self.resize), interpolation=Image.BILINEAR)
        gt = F.resize(gt, size=(self.resize, 2*self.resize), interpolation=Image.NEAREST)
        gt = np.asarray(gt).copy()
        idx3 = np.isin(gt, 3)
        idx4 = np.isin(gt, 4)
        gt[idx3] = 0
        gt[idx4] = 0
        # params = [params[0], params[1]]
        hflip_input = np.random.uniform(0.0, 1.0) > 0.5 and self.flip_on
        if idx not in self.valid_idx and hflip_input:
            image, gt = F.hflip(image), F.hflip(gt)
            line_lst = mirror_list(line_lst)

            idx1 = np.isin(gt, 1)
            idx2 = np.isin(gt, 2)
            gt[idx1] = 2
            gt[idx2] = 1
            params = [params[1], params[0], params[3], params[2]]
            params = np.array(params)
            params = -params
            params[:, -1] = 1+params[:, -1]


        gt = Image.fromarray(gt)
        params = torch.from_numpy(np.array(params)).float()
        image, gt = self.totensor(image).float(), (self.totensor(gt)*255).long()

        y_val = gt.nonzero()[0, 1]
        horizon = torch.zeros(gt.size(1))
        horizon[0:y_val] = 1
        line_lst = np.array(line_lst[3:7])
        line_lst = torch.from_numpy(np.array(line_lst + 1))
        line_lst = line_lst.long()
        horizon = horizon.float()

        if idx in self.valid_idx:
            index = self.valid_idx.index(idx)
            return image, gt, params, idx, line_lst, horizon, index
        return image, gt, params, idx, line_lst, horizon


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


class Scale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, type_dataset, interpolation=Image.BILINEAR):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.interpolation = interpolation
        self.type = type_dataset

    def __call__(self, sample):
        image = sample['image']

        h, w = image.size
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = image.resize((new_h, new_w), Image.BILINEAR)
        if self.type == 'train':
            gt = sample['ground truth']
            gt = gt.resize((new_h, new_w), Image.NEAREST)
            return {'image': (img), 'ground truth': (gt)}
        else:
            return {'image': np.asarray(img)}


class Crop(object):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    def __init__(self, start_y, type_dataset):
        self.i = start_y
        self.type = type_dataset

    def __call__(self, sample):

        image = sample['image'] 
        image = image.crop((0, self.i, 1280, 720))
        if self.type == 'train':
            gt = sample['ground truth']
            gt = gt.crop((0, self.i, 1280, 720))
            return {'image': image, 'ground truth': gt}
        else:
            return {'image': image}


class RandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size=(280, 560), padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        img, gt = sample['image'], sample['ground truth']
        
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            gt = ImageOps.expand(gt, border=self.padding, fill=0)

        i, j, h, w = self.get_params(img, self.size)
        img = img.crop((j, i, j + w, i + h))
        gt = gt.crop((j, i, j + w, i + h))

        return {'image': img, 'ground truth': gt}


def plot_dataloader_batch(batch):
    images_batch = batch['image']
    plt.figure()
    grid_im = utils.make_grid(images_batch)
    plt.imshow(grid_im.numpy().transpose((1, 2, 0)))


def get_loader(num_train, json_file, image_dir, gt_dir, flip_on, batch_size,
               shuffle, num_workers, end_to_end, resize, split_percentage=0.2):
    '''
    Splits dataset in training and validation set and creates dataloaders
    '''
    indices = list(range(num_train))
    split = int(np.floor(split_percentage*num_train))

    if shuffle is True:
        np.random.seed(num_train)
        np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_idx = train_idx[0:len(train_idx)//batch_size*batch_size]
    valid_idx = valid_idx[0:len(valid_idx)//batch_size*batch_size]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

    transformed_dataset = LaneDataset(end_to_end=end_to_end,
                                      valid_idx=valid_idx,
                                      json_file=json_file,
                                      image_dir=image_dir,
                                      gt_dir=gt_dir,
                                      flip_on=flip_on,
                                      resize=resize)

    train_loader = DataLoader(transformed_dataset,
                              batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True) #, collate_fn=my_collate)

    valid_loader = DataLoader(transformed_dataset,
                              batch_size=batch_size, sampler=valid_sampler,
                              num_workers=num_workers, pin_memory=True) #collate_fn=my_collate)

    return train_loader, valid_loader, valid_idx


def my_collate(batch):
    batch = [dict for dict in batch if dict['4_lanes'] is True]
    return default_collate(batch)


def write_lsq_results(src_file, dst_file, nclasses, all_branches_ready, 
                      horizon_on, resize, no_ortho, calc_intersection=False, 
                      draw_image=False, path_test_set = '../../../', test_phase=False):
    '''
    Computes json file with point coordinates for every lane
    '''
    factor = 640/resize
    y_start = 0.3
    y_stop = 1
    src = np.float32([[0.45,y_start],[0.55, y_start],[0.1,y_stop],[0.9, y_stop]])
    dst = np.float32([[0.45, y_start],[0.55, y_start],[0.45, y_stop],[0.55,y_stop]])
    M = cv2.getPerspectiveTransform(src,dst)
    M_inv = cv2.getPerspectiveTransform(dst,src)
    lines = [json.loads(line) for line in open(src_file).readlines()]
    with open(dst_file, 'w') as jsonFile:
        for i, line in enumerate(lines):
            h_samples = line['h_samples']
            y_orig = np.array(h_samples)
            y_d = (np.array(h_samples)-80)/639
            y_prime = ((M[1][1]*y_d + M[1][2])/(M[2][1]*y_d+M[2][2]))
            y_eval = (1-y_prime)
            lanes_json = np.full((nclasses,len(h_samples)), -2)
            lanes = line["lanes"]
            params = line["params"]
            line_id = line["line_id"]
            horizon = line["horizon_est"]
            
            if draw_image and i%50==0:
                path = path_test_set + 'test_set/' + line["raw_file"]
                img = plt.imread(path)
                img = np.asarray(img)
            
            if calc_intersection:
                maxima = instersection_points(params, M_inv, resize)
                
            no_left_line = True if line_id[0] == 0 else False
            no_right_line = True if line_id[3] == 0 else False
            for j in range(len(params)):
                if test_phase:
                    lane = lanes
                else:
                    lane = lanes[j]
                if all_branches_ready:
                    if (j==2 and no_left_line) or (j==3 and no_right_line):
                        continue
                else:
                    zipped_y_vals = [(x,y) for x,y in zip(lane, h_samples) if x!=-2]
                    if len(zipped_y_vals) == 0:
                        continue
                
                h = [y for x,y in zip(lane, h_samples) if x!=-2]
                if len(h) == 0:
                    minimum, maximum = 250, 710
                else:
                    minimum, maximum = np.min(h), np.max(h) 
                if all_branches_ready and horizon_on:
                    minimum = sum(horizon)*factor+80
                    if calc_intersection:
                        maximum = maxima[j]*factor+84
                params_j = [0]*(3-len(params[j])) + params[j]
                a,b,c = params_j
                
                if not no_ortho:
                    x_new = (a*y_eval**2 + b*y_eval + c)
                    x_new, y_new = homogenous_transformation(M_inv, x_new, y_prime)
                else:
                    y_new = 1-y_d
                    x_new = (a*y_new**2 + b*y_new + c)
                x_new, y_new = x_new*1279, y_new*639+80 
                x_new = np.int_(np.round(x_new))
                x_new, y_new = zip(*[(x,y) if y >= max(210,minimum) and y <= maximum else (-2,y) 
                                for x,y in zip(x_new, y_orig)])
                
                lanes_json[j] = list(x_new)
                
                if draw_image and i%50==0:
                    pt = [(xcord, ycord) for (xcord, ycord) in zip(x_new, y_new) if xcord!=-2]
                    for idx in pt:
                        cv2.circle(img, idx, radius=4, thickness=-1, color=(255, 0, 0))
            if draw_image and i%50==0:
                img = Image.fromarray(img)
                img.save('../Evaluate/Results/{}.png'.format(i))
            json_line = line
            json_line["run_time"] = 20
            json_line["lanes"] = lanes_json.tolist()
            json.dump(json_line, jsonFile)
            jsonFile.write('\n' )


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
    file1 = "Labels/Curve_parameters.json"
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


def get_testloader(json_file, transform, batch_size, shuffle, num_workers, path):
    transformed_dataset = LaneTestSet(json_file=json_file,
                                      path=path,
                                      transform=transform)

    data_loader = DataLoader(dataset=transformed_dataset, 
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          num_workers=num_workers)
    return data_loader
