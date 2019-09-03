"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import ceil
import cv2
import Networks
from Networks.gels import GELS
from Networks.utils import get_homography


def square_tensor(x):
    return x**2


def return_tensor(x):
    return x


def activation_layer(activation='square', no_cuda=False):
    place_cuda = True
    if activation == 'sigmoid':
        layer = nn.Sigmoid()
    elif activation == 'relu':
        layer = nn.ReLU()
    elif activation == 'softplus':
        layer = nn.Softplus()
    elif activation == 'square':
        layer = square_tensor
        place_cuda = False
    elif activation == 'abs':
        layer = torch.abs
        place_cuda = False
    elif activation == 'none':
        layer = return_tensor
    else:
        raise NotImplementedError('Activation type: {} is not implemented'.format(activation))
    if not no_cuda and place_cuda:
        layer = layer.cuda()
    return layer


def ProjectiveGridGenerator(size, theta, no_cuda):
    # Compute base grid (in img space)
    N, C, H, W = size
    linear_points_W = torch.linspace(0, W - 1, W)
    linear_points_H = torch.linspace(0, H - 1, H)

    base_grid = theta.new(N, H, W, 3)
    base_grid[:, :, :, 0] = torch.ger(
            torch.ones(H), linear_points_W).expand_as(base_grid[:, :, :, 0])
    base_grid[:, :, :, 1] = torch.ger(
            linear_points_H, torch.ones(W)).expand_as(base_grid[:, :, :, 1])
    base_grid[:, :, :, 2] = 1

    # Transform base grid (to homography space)
    grid = torch.bmm(base_grid.view(N, H * W, 3), theta.transpose(1, 2))
    grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:])
    if not no_cuda:
        grid = grid.cuda()
    return grid



class Weighted_least_squares(nn.Module):
    def __init__(self, size, nclasses, order, no_cuda, reg_ls=0, use_cholesky=False):
        super().__init__()
        N, C, self.H, W = size
        self.nclasses = nclasses
        self.tensor_ones = torch.ones(N, self.H*W, 1).float()
        self.order = order
        self.reg_ls = reg_ls*torch.eye(order+1)
        self.use_cholesky = use_cholesky
        if not no_cuda:
            self.reg_ls = self.reg_ls.cuda()
            self.tensor_ones = self.tensor_ones.cuda()

    def forward(self, W, grid):
        beta2, beta3 = None, None

        # Prepare x-y grid
        W = W.view(-1, self.nclasses, grid.size(1))
        bs = W.size(0)
        grid = grid[:bs]
        tensor_ones = self.tensor_ones[:bs]
        x_map = grid[:, :, 0].unsqueeze(2)
        y_map = (255 - grid[:, :, 1]).unsqueeze(2) # No pixel coordinates here!

        # Compute matrix Y (to solve system W*Y*beta = W*X)
        if self.order == 0:
            Y = self.tensor_ones
        elif self.order == 1:
            Y = torch.cat((y_map, tensor_ones), 2)
        elif self.order == 2:
            Y = torch.cat((y_map**2, y_map, tensor_ones), 2)
        elif self.order == 3:
            Y = torch.cat((y_map**3, y_map**2, y_map, tensor_ones), 2)
        else:
            raise NotImplementedError(
                    'Requested order {} for polynomial fit is not implemented'.format(self.order))

        # Left egoline
        W0 = W[:, 0, :].unsqueeze(2)
        Y0 = torch.mul(W0, Y)
        if not self.use_cholesky:
            Z = torch.bmm(Y0.transpose(1, 2), Y0) + self.reg_ls
            Z_inv = torch.inverse(Z)
            X = torch.bmm(Y0.transpose(1, 2), torch.mul(W0, x_map))
            beta0 = torch.bmm(Z_inv, X)
        else:
            beta0 = GELS.apply(Y0, torch.mul(W0, x_map))

        # Right egoline
        W1 = W[:, 1, :].unsqueeze(2)
        Y1 = torch.mul(W1, Y)
        if not self.use_cholesky:
            Z = torch.bmm(Y1.transpose(1, 2), Y1) + self.reg_ls
            Z_inv = torch.inverse(Z)
            X = torch.bmm(Y1.transpose(1, 2), torch.mul(W1, x_map))
            beta1 = torch.bmm(Z_inv, X)
        else:
            beta1 = GELS.apply(Y1, torch.mul(W1, x_map))

        # Further lane lines
        if self.nclasses > 3:
            W2 = W[:, 2, :].unsqueeze(2)
            Y2 = torch.mul(W2, Y)
            if not self.use_cholesky:
                Z = torch.bmm(Y2.transpose(1, 2), Y2) + self.reg_ls
                Z_inv = torch.inverse(Z)
                X = torch.bmm(Y2.transpose(1, 2), torch.mul(W2, x_map))
                beta2 = torch.bmm(Z_inv, X)
            else:
                beta2 = GELS.apply(Y2, torch.mul(W2, x_map))
            beta2 = beta2.double()
            W3 = W[:, 3, :].unsqueeze(2)
            Y3 = torch.mul(W3, Y)
            if not self.use_cholesky:
                Z = torch.bmm(Y3.transpose(1, 2), Y3) + self.reg_ls
                Z_inv = torch.inverse(Z)
                X = torch.bmm(Y3.transpose(1, 2), torch.mul(W3, x_map))
                beta3 = torch.bmm(Z_inv, X)
            else:
                beta3 = GELS.apply(Y3, torch.mul(W3, x_map))
            beta3 = beta3.double()

        return beta0.double(), beta1.double(), beta2, beta3


class Classification(nn.Module):
    def __init__(self, class_type, size, channels_in, resize):
        super().__init__()
        self.class_type = class_type
        filter_size = 1
        pad = (filter_size-1)//2
        self.conv1 = nn.Conv2d(channels_in, 128, filter_size,
           stride=1, padding=pad, bias=True)
        self.conv1_bn = nn.BatchNorm2d(128)
        
        filter_size = 3
        pad = (filter_size-1)//2
        self.conv2 = nn.Conv2d(128, 128, filter_size,
           stride=1, padding=pad, bias=True)
        self.conv2_bn = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 64, filter_size,
           stride=1, padding=pad, bias=True)
        self.conv3_bn = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 64, filter_size,
           stride=1, padding=pad, bias=True)
        self.conv4_bn = nn.BatchNorm2d(64)
        
        rows, cols = size
        self.avgpool = nn.AvgPool2d((1,cols))
        self.maxpool = nn.MaxPool2d((2, 2), stride = 2)
        
        if class_type == 'line':
            self.fully_connected1 = nn.Linear(64*rows*cols//4, 128)
            self.fully_connected_line1 = nn.Linear(128, 4)
        else:
            self.fully_connected_horizon = nn.Linear(64*rows, resize)
            
    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        if self.class_type == 'line':
            x = self.maxpool(x)
        else:
            x = self.avgpool(x)
        x = x.view(x.size()[0],-1)
        batch_size = x.size(0)
        if self.class_type == 'line':
            x = F.relu(self.fully_connected1(x))
            x = self.fully_connected_line1(x)
        else:
            x = self.fully_connected_horizon(x)
        return x


class Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.nclasses = args.nclasses

        # define sizes and perspective transformation
        resize = args.resize
        size = torch.Size([args.batch_size, args.nclasses, args.resize, 2*args.resize])
        M, _ = get_homography(args.resize, args.no_mapping)
        M = torch.from_numpy(M).unsqueeze_(0).expand([args.batch_size, 3, 3]).float()

        # Define network
        out_channels = args.nclasses + int(not args.end_to_end)

        self.net = Networks.define_model(mod=args.mod, layers=args.layers, 
                                         in_channels=args.channels_in,
                                         out_channels=out_channels, 
                                         pretrained=args.pretrained, pool=args.pool)
        # Init activation
        self.activation = activation_layer(args.activation_layer, args.no_cuda)
        # Init grid generator
        self.grid = ProjectiveGridGenerator(size, M, args.no_cuda)
        # Init LS layer
        self.ls_layer = Weighted_least_squares(size, args.nclasses, args.order, 
                args.no_cuda, args.reg_ls, args.use_cholesky)

        # mask configuration
        zero_rows = ceil(args.resize*args.mask_percentage)
        self.idx_row = torch.linspace(0, zero_rows-1, zero_rows).long()
        n_row = 13
        self.idx_col1 = Variable(torch.linspace(1, n_row, n_row+1).long())
        self.idx_col2 = Variable(torch.linspace(0, n_row, n_row+1).long())+2*resize-(n_row+1)
        idx_mask = (np.arange(resize)[:, None] < np.arange(2*resize)-(resize+10))*1
        idx_mask = np.flip(idx_mask, 1).copy() + idx_mask
        self.idx_mask = Variable(torch.from_numpy(idx_mask)) \
                .type(torch.ByteTensor).expand(
                        args.batch_size, args.nclasses, resize, 2*resize)

        self.end_to_end = args.end_to_end
        self.pretrained = args.pretrained
        self.classification_branch = args.clas
        if self.classification_branch:
            size_enc = (32, 64)
            chan = 128
            self.line_classification = Classification('line', size=size_enc, 
                    channels_in=chan, resize=resize)
            self.horizon_estimation = Classification('horizon', size=size_enc, 
                    channels_in=chan, resize=resize)

        # Place on GPU if specified
        if not args.no_cuda:
            self.idx_row = self.idx_row.cuda()
            self.idx_col1 = self.idx_col1.cuda()
            self.idx_col2 = self.idx_col2.cuda()
            self.idx_mask = self.idx_mask.cuda()
            if self.classification_branch:
                self.line_classification = self.line_classification.cuda()
                self.horizon_estimation = self.horizon_estimation.cuda()

    def forward(self, input, gt_line, end_to_end, early_return=False, gt=None):
        # 0. Init variables
        line, horizon = None, None

        # 1. Push batch trough network
        shared_encoder, output, output_seg = self.net(input, end_to_end*self.pretrained)
        if early_return:
            return output

        # 2. use activation function
        if not end_to_end:
            activated = output.detach()
            _, activated = torch.max(activated, 1)

            activated = activated.float()
            if self.nclasses < 3:
                left = activated*(activated == 1).float()
                right = activated*(activated == 2).float()
                activated = torch.stack((left, right), 1)
            else:
                left1 = activated*(activated == 1).float()
                right1 = activated*(activated == 2).float()
                left2 = activated*(activated == 3).float()
                right2 = activated*(activated == 4).float()
                activated = torch.stack((left1, right1, left2, right2), 1)
        else:
            activated = self.activation(output)
            if self.classification_branch:
                line = self.line_classification(shared_encoder)
                horizon = self.horizon_estimation(shared_encoder)

        # 3. use mask
        masked = activated.index_fill(2, self.idx_row, 0)
        # trapezium mask not needed for only two lanes
        # Makes convergence easier for lane lines further away
        # output = output.index_fill(3,self.idx_col1,0)
        # output = output.index_fill(3,self.idx_col2,0)
        # output = output.masked_fill(self.idx_mask,0)

        # Prevent singular matrix
        if gt_line.sum() != 0 and end_to_end == False:
            gt_mask = gt_line[:, :, None, None].byte().expand_as(masked)
            masked[gt_mask] = masked[0, 0].unsqueeze(0).repeat(gt_line.sum().item(), 1, 1).view(-1)

        # 4. Least squares layer
        beta0, beta1, beta2, beta3 = self.ls_layer(masked, self.grid)
        return beta0, beta1, beta2, beta3, masked, output, line, horizon, output_seg
