#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import ceil
import cv2
import Networks


def Init_Projective_transform(nclasses, batch_size, resize):
    # M_orig: unnormalized Transformation matrix
    # M: normalized transformation matrix
    # M_inv: Inverted normalized transformation matrix --> Needed for grid sample
    # original aspect ratio: 720x1280 --> after 80 rows cropped: 640x1280 --> after resize: 256x512 (default) or resize x 2*resize (in general)
    size = torch.Size([batch_size, nclasses, resize, 2*resize])
    y_start = 0.3
    y_stop = 1
    xd1, xd2, xd3, xd4 = 0.45, 0.55, 0.45, 0.55
    src = np.float32([[0.45, y_start], [0.55, y_start], [0.1, y_stop], [0.9, y_stop]])
    dst = np.float32([[xd3, y_start], [xd4, y_start], [xd1, y_stop], [xd2, y_stop]])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    M = torch.from_numpy(M).unsqueeze_(0).expand([batch_size, 3, 3]).type(torch.FloatTensor)
    M_inv = torch.from_numpy(M_inv).unsqueeze_(0).expand([batch_size, 3, 3]).type(torch.FloatTensor)
    return size, M, M_inv


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


class ProjectiveGridGenerator(nn.Module):
    def __init__(self, size, theta, no_cuda):
        super().__init__()
        self.N, self.C, self.H, self.W = size
        linear_points_W = torch.linspace(0, 1 - 1/self.W, self.W)
        linear_points_H = torch.linspace(0, 1 - 1/self.H, self.H)

        self.base_grid = theta.new(self.N, self.H, self.W, 3)
        self.base_grid[:, :, :, 0] = torch.ger(
                torch.ones(self.H), linear_points_W).expand_as(self.base_grid[:, :, :, 0])
        self.base_grid[:, :, :, 1] = torch.ger(
                linear_points_H, torch.ones(self.W)).expand_as(self.base_grid[:, :, :, 1])
        self.base_grid[:, :, :, 2] = 1

        self.base_grid = Variable(self.base_grid)
        if not no_cuda:
            self.base_grid = self.base_grid.cuda()

    def forward(self, theta):
        grid = torch.bmm(self.base_grid.view(self.N, self.H * self.W, 3), theta.transpose(1, 2))
        grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:])
        return grid


class Weighted_least_squares(nn.Module):
    def __init__(self, size, nclasses, order, no_cuda, reg_ls=0, use_cholesky=False):
        super().__init__()
        N, C, self.H, W = size
        self.nclasses = nclasses
        self.tensor_ones = Variable(torch.ones(N, self.H*W, 1))
        self.order = order
        self.reg_ls = Variable(reg_ls*torch.eye(order+1))
        self.use_cholesky = use_cholesky
        if not no_cuda:
            self.reg_ls = self.reg_ls.cuda()
            self.tensor_ones = self.tensor_ones.cuda()

    def forward(self, W, grid):
        beta2, beta3 = None, None

        W = W.view(-1, self.nclasses, grid.size(1))
        W0 = W[:, 0, :].unsqueeze(2)
        x_map = grid[:, :, 0].unsqueeze(2)
        y_map = ((1)-grid[:, :, 1]).unsqueeze(2)
        if self.order == 0:
            Y = self.tensor_ones
        elif self.order == 1:
            Y = torch.cat((y_map, self.tensor_ones), 2)
        elif self.order == 2:
            Y = torch.cat(((y_map**2), y_map, self.tensor_ones), 2)
        else:
            raise NotImplementedError(
                    'Requested order {} for polynomial fit is not implemented'.format(self.order))

#        Y = Y[0:W.size(0),:,:]
#        x_map = x_map[0:W.size(0),:,:]
        Y0 = torch.mul(W0, Y)
        Z = torch.bmm(Y0.transpose(1, 2), Y0) + self.reg_ls

        if not self.use_cholesky:
            # Z_inv = [torch.inverse(matrix) for matrix in torch.unbind(Z)]
            # Z_inv = torch.stack(Z_inv)
            Z_inv = torch.inverse(Z)
            X = torch.bmm(Y0.transpose(1, 2), torch.mul(W0, x_map))
            beta0 = torch.bmm(Z_inv, X)
        else:
            # cholesky
            # TODO check this
            beta0 = []
            X = torch.bmm(Y0.transpose(1, 2), torch.mul(W0, x_map))
            for image, rhs in zip(torch.unbind(Z), torch.unbind(X)):
                R = torch.potrf(image)
                opl = torch.trtrs(rhs, R.transpose(0, 1))
                beta0.append(torch.trtrs(opl[0], R, upper=False)[0])
            beta0 = torch.cat((beta0), 1).transpose(0, 1).unsqueeze(2)

        W1 = W[:, 1, :].unsqueeze(2)
        Y1 = torch.mul(W1, Y)
        Z = torch.bmm(Y1.transpose(1, 2), Y1) + self.reg_ls
        # TODO : use torch.inverse over batch (is possible since pytorch 1.0.0)
        # Z_inv = [torch.inverse(matrix) for matrix in torch.unbind(Z)]
        # Z_inv = torch.stack(Z_inv)
        Z_inv = torch.inverse(Z)
        X = torch.bmm(Y1.transpose(1, 2), torch.mul(W1, x_map))
        beta1 = torch.bmm(Z_inv, X)

        if self.nclasses > 3:
            W2 = W[:, 2, :].unsqueeze(2)
            Y2 = torch.mul(W2, Y)
            Z = torch.bmm(Y2.transpose(1, 2), Y2) + self.reg_ls
            Z_inv = torch.inverse(Z)
            X = torch.bmm(Y2.transpose(1, 2), torch.mul(W2, x_map))
            beta2 = torch.bmm(Z_inv, X)

            W3 = W[:, 3, :].unsqueeze(2)
            Y3 = torch.mul(W3, Y)
            Z = torch.bmm(Y3.transpose(1, 2), Y3) + self.reg_ls
            Z_inv = torch.inverse(Z)
            X = torch.bmm(Y3.transpose(1, 2), torch.mul(W3, x_map))
            beta3 = torch.bmm(Z_inv, X)

        return beta0, beta1, beta2, beta3


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
            self.fully_connected_line1 = nn.Linear(128, 3)
            self.fully_connected_line2 = nn.Linear(128, 3)
            self.fully_connected_line3 = nn.Linear(128, 3)
            self.fully_connected_line4 = nn.Linear(128, 3)
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
            x1 = self.fully_connected_line1(x).view(batch_size,3,1,1)
            x2 = self.fully_connected_line2(x).view(batch_size,3,1,1)
            x3 = self.fully_connected_line3(x).view(batch_size,3,1,1)
            x4 = self.fully_connected_line4(x).view(batch_size,3,1,1)
            
            x = torch.cat((x1, x2, x3, x4), 2).squeeze(3)
        else:
            x = self.fully_connected_horizon(x)
        return x


class Net(nn.Module):
    def __init__(self, args):
        super().__init__()

        # define sizes and perspective transformation
        resize = args.resize
        size = torch.Size([args.batch_size, args.nclasses, args.resize, 2*args.resize])
        size, M, M_inv = Init_Projective_transform(args.nclasses, args.batch_size, args.resize)
        self.M = M

        # Define network
        out_channels = args.nclasses + int(not args.end_to_end)

        self.net = Networks.define_model(mod=args.mod, layers=args.layers, 
                                         in_channels=args.channels_in,
                                         out_channels=out_channels, 
                                         pretrained=args.pretrained, pool=args.pool)
        # Init activation
        self.activation = activation_layer(args.activation_layer, args.no_cuda)
        # Init grid generator
        self.project_layer = ProjectiveGridGenerator(size, M, args.no_cuda)
        # Init LS layer
        self.ls_layer = Weighted_least_squares(size, args.nclasses, args.order, 
                args.no_cuda, args.reg_ls, args.use_cholesky)

        # mask configuration
        zero_rows = ceil(args.resize*args.mask_percentage)
        self.idx_row = Variable(torch.linspace(0, zero_rows-1, zero_rows).long())
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
            self.M = self.M.cuda()
            self.idx_row = self.idx_row.cuda()
            self.idx_col1 = self.idx_col1.cuda()
            self.idx_col2 = self.idx_col2.cuda()
            self.idx_mask = self.idx_mask.cuda()
            if self.classification_branch:
                self.line_classification = self.line_classification.cuda()
                self.horizon_estimation = self.horizon_estimation.cuda()

    def forward(self, input, end_to_end):
        # 0. Init variables
        line, horizon = None, None

        # 1. Push batch trough network
        shared_encoder, output = self.net(input, end_to_end*self.pretrained)

        # 2. use activation function
        # if not self.end_to_end:
            # output = output.detach()
        # activated = output[:, 1:, :, :]
        # activated = torch.clamp(activated, min=0)
        if not end_to_end:
            activated = output.detach()
            _, activated = torch.max(activated, 1)
            activated = activated.float()
            left = activated*(activated == 1).float()
            right = activated*(activated == 2).float()
            activated = torch.stack((left, right), 1)
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

        # 4. Least squares layer
        grid = self.project_layer(self.M)
        beta0, beta1, beta2, beta3 = self.ls_layer(masked, grid)
        return beta0, beta1, beta2, beta3, masked, self.M, output, line, horizon
