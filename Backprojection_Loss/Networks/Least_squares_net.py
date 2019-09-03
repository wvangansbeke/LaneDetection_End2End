"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler
from torch.autograd import Variable
from Networks.segnet_long import SegNet
from Networks.Enet_vanilla import ENet
from Networks.unet import UNet, _DecoderBlock
from Networks.vision import ResNet18

import cv2
from math import ceil

def Init_Projective_transform(nclasses, batch_size, resize, grid_sample, sample_factor):
    #M_orig: unnormalized Transformation matrix
    #M: normalized transformation matrix
    #M_inv: Inverted normalized transformation matrix --> Needed for grid sample
    #original aspect ratio: 720x1280 --> after 80 rows cropped: 640x1280 --> after resize: 320x640 (default) or resize x 2*resize (in general)
    if not grid_sample:
        size = torch.Size([batch_size, nclasses, resize, 2*resize])
    else: 
        size = torch.Size([batch_size, nclasses, resize//sample_factor+1, 2*resize//sample_factor+1])
    y_start = 0.3
    y_stop = 1
    xd1, xd2, xd3, xd4 = 0.45, 0.55, 0.45, 0.55
    src = np.float32([[0.45,y_start],[0.55, y_start],[0.1,y_stop],[0.9, y_stop]])
    dst = np.float32([[xd3, y_start],[xd4, y_start],[xd1, y_stop],[xd2, y_stop]])
    M = cv2.getPerspectiveTransform(src,dst)
    M_inv = cv2.getPerspectiveTransform(dst,src)
    M = torch.from_numpy(M).unsqueeze_(0).expand([batch_size,3,3]).type(torch.FloatTensor)
    M_inv = torch.from_numpy(M_inv).unsqueeze_(0).expand([batch_size,3,3]).type(torch.FloatTensor)
    return size, M, M_inv

class resnet_block(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, encode=True):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
            
    def forward(self, x):
        shortcut = self.conv_shortcut(x)
        
        x = F.relu(self.bn1(x))
        x = self.conv1(x)
        x = F.relu(self.bn2(x))
        x = self.conv2(x)
        
        x = x + shortcut
#        x = F.relu(x) #W relu toegevoegd
        
        return x

class simple_net(nn.Module):
    def __init__(self, nclasses):
        super().__init__()
#        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3,
#                               bias=False)
#        
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.resnet_block1 = resnet_block(32, 32, 1)
        self.pool = nn.MaxPool2d(2, (2,2))
        self.resnet_block2 = resnet_block(32, 64, 1)
        self.resnet_block3 = resnet_block(64, 128, 2) # dilation 2
        self.resnet_block4 = resnet_block(128, 256, 4) # dilation 4
        self.resnet_block5 = resnet_block(256, 128, 2) # dilation 2
        self.resnet_block6 = resnet_block(128, 64, 1)
        self.upsample = nn.ConvTranspose2d(64, 64,kernel_size=2, stride=2)
        self.resnet_block7 = resnet_block(64, 32, 1)
        self.conv_out = nn.Conv2d(32, nclasses, 1, stride=1, padding=0, bias=True)
#        self.conv_out = nn.ConvTranspose2d(32, nclasses,kernel_size=2, stride=2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet_block1(x)
#        x = self.pool(x)
        x = self.resnet_block2(x)
        x = self.resnet_block3(x)
        x = self.resnet_block4(x)
        output_encoder = x
        
        x = self.resnet_block5(x)
        x = self.resnet_block6(x)
#        x = self.upsample(x)
        x = self.resnet_block7(x)
        x = self.conv_out(x)
        
        return x, output_encoder
    
class Classification_old(nn.Module):
    def __init__(self, class_type, size, resize=320):
        super().__init__()
        self.class_type = class_type
        
        self.conv1 = nn.Conv2d(512, 256, 3,
           stride=1, padding=1, bias=True)
        self.conv1_bn = nn.BatchNorm2d(256)
        
        self.conv2 = nn.Conv2d(256, 128, 3,
           stride=1, padding=1, bias=True)
        self.conv2_bn = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 64, 3,
           stride=1, padding=1, bias=True)
        self.conv3_bn = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 64, 3,
           stride=1, padding=1, bias=True)
        self.conv4_bn = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 32, 3,
           stride=1, padding=1, bias=True)
        self.conv5_bn = nn.BatchNorm2d(32)
        
        self.pool1 = nn.MaxPool2d((2, 2), stride = 2)
        self.pool2 = nn.MaxPool2d((2, 2), stride = 2)
        rows, cols = size
        self.fully_connected1 = nn.Linear(32*rows*cols//4, 1024)
#        self.fully_connected1 = nn.Linear(3648, 1024)
        if class_type == 'line':
            self.fully_connected2 = nn.Linear(1024, 128)
            self.fully_connected_line1 = nn.Linear(128, 3)
            self.fully_connected_line2 = nn.Linear(128, 3)
            self.fully_connected_line3 = nn.Linear(128, 3)
            self.fully_connected_line4 = nn.Linear(128, 3)
        else:
            self.fully_connected_horizon = nn.Linear(1024, resize)
            
    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
#        x = self.pool1(x)
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = self.pool1(x)
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fully_connected1(x))
        
        batch_size = x.size(0)
        if self.class_type == 'line':
            x = F.relu(self.fully_connected2(x))
            x1 = self.fully_connected_line1(x).view(batch_size,3,1,1)
            x2 = self.fully_connected_line2(x).view(batch_size,3,1,1)
            x3 = self.fully_connected_line3(x).view(batch_size,3,1,1)
            x4 = self.fully_connected_line4(x).view(batch_size,3,1,1)
            
            x = torch.cat((x1, x2, x3, x4), 2)
            
        else:
            x = self.fully_connected_horizon(x)
        return x

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
            
            x = torch.cat((x1, x2, x3, x4), 2)
        else:
            x = self.fully_connected_horizon(x)
        return x


class DLT(nn.Module):
    def __init__(self, batch_size, cuda, size, channels_in):
        super().__init__()
        self.activation = nn.Tanh()
        self.spatial_trans = Spatial_transformer_net(size, channels_in)
        self.xs1, self.xs2, self.xs3, self.xs4 = 0.1, 0.9, 0.45, 0.55
        self.ys1, self.ys2 = 1, 0.3
        self.xd1, self.xd2, self.xd3, self.xd4 = 0.45, 0.55, 0.45, 0.55
        self.yd1, self.yd2 = 1, 0.3
        A = torch.FloatTensor([[0,0,0,-self.ys1,-1,self.ys1*self.yd1],
                                        [self.xs1,self.ys1,1,0,0,0],
                                        [self.xs2,self.ys1,1,0,0,0],
                                        [0,0,0,-self.ys2,-1,0],
                                        [self.xs3,self.ys2,1,0,0,0],
                                        [self.xs4,self.ys2,1,0,0,0]])
        
        B = torch.FloatTensor([[-self.yd1],
                               [0],
                               [0],
                               [0],
                               [0],
                               [0]])
        self.A = A.expand(batch_size,6,6)
        self.B = B.expand(batch_size,6,1)
        self.zeros = Variable(torch.zeros(batch_size,1,1))
        self.ones = Variable(torch.ones(batch_size,1,1))
        if cuda: 
            self.activation = self.activation.cuda()
            self.A = self.A.cuda()
            self.B = self.B.cuda()
            self.zeros = self.zeros.cuda()
            self.ones = self.ones.cuda()
            self.spatial_trans = self.spatial_trans.cuda()
    def forward(self, output_encoder):
        x = self.spatial_trans(output_encoder)
        x = self.activation(x)/16
        A = Variable(self.A.clone())
        B = Variable(self.B.clone())
        A[:,1,5] = -self.ys1*(self.xd1+x[:,0])
        A[:,2,5] = -self.ys1*(self.xd2+x[:,1])
        A[:,3,5] =  self.ys2*(self.yd2+x[:,2])
        A[:,4,5] = -self.ys2*(self.xd3+x[:,0])
        A[:,5,5] = -self.ys2*(self.xd4+x[:,1])
        
        B[:,1,0] =   self.xd1+x[:,0]
        B[:,2,0] =   self.xd2+x[:,1]
        B[:,3,0] = -(self.yd2+x[:,2])
        B[:,4,0] =   self.xd3+x[:,0]
        B[:,5,0] =   self.xd4+x[:,1]
        
        A_prime = torch.bmm(A.transpose(1,2), A)
        B_prime = torch.bmm(A.transpose(1,2), B)
        
        h = torch.stack([torch.gesv(b, a)[0] for b, a in zip(torch.unbind(B_prime), torch.unbind(A_prime))])
        h = torch.cat((h[:,0:3,:],self.zeros,h[:,3:5,:],self.zeros,h[:,5:6,:],self.ones),1)
        h = h.view(-1,3,3)
#        x.register_hook(save_grad('x'))
        return h, x
    
class Spatial_transformer_net(nn.Module):
    def __init__(self, size, channels_in):
        super().__init__()
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
        self.maxpool = nn.MaxPool2d((2, 2), stride = 2)
        
        self.conv5 = nn.Conv2d(64, 64, filter_size,
           stride=1, padding=pad, bias=True)
        self.conv4_bn = nn.BatchNorm2d(64)
        
        self.fully_connected1 = nn.Linear(64*rows*cols//4, 128)
        self.fully_connected2 = nn.Linear(128, 3)
        
        self.fully_connected2.weight.data.fill_(0)
        self.fully_connected2.bias.data = torch.FloatTensor([0, 0, 0])
   
    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = self.maxpool(x)
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fully_connected1(x))
        x = self.fully_connected2(x)
        return x


class Proxy_branch_segmentation(nn.Module):
    def __init__(self, channels_in, resize, nclasses):
        super().__init__()
        kernel_size = 3
        padding = (kernel_size-1)//2
        self.dec4 = _DecoderBlock(channels_in, 256, 256, pad=padding)
        self.dec3 = _DecoderBlock(256, 128, 128, pad=padding)
        self.dec2 = _DecoderBlock(128, 64, 64, pad=padding)
        self.final = nn.Conv2d(64, nclasses+1, kernel_size=1)
        self.resize = resize

    def forward(self, x):
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.final(x)
        x = F.upsample(x, (self.resize, 2*self.resize), mode='bilinear')
        return x
        
    
class ProjectiveGridGenerator(nn.Module):
    def __init__(self, size, theta, cuda):
        super().__init__()
        self.N, self.C, self.H, self.W = size
        linear_points_W = torch.linspace(0, 1, self.W)
        linear_points_H = torch.linspace(0, 1, self.H)

        self.base_grid = theta.new(self.N, self.H, self.W, 3)
        self.base_grid[:, :, :, 0] = torch.ger(torch.ones(self.H), linear_points_W).expand_as(self.base_grid[:, :, :, 0])
        self.base_grid[:, :, :, 1] = torch.ger(linear_points_H, torch.ones(self.W)).expand_as(self.base_grid[:, :, :, 1])
        self.base_grid[:, :, :, 2] = 1
        
        self.base_grid = Variable(self.base_grid)
        if cuda:
            self.base_grid = self.base_grid.cuda()
            
    def forward(self, theta, no_ortho_view):
        if no_ortho_view:
            return self.base_grid.view(self.N, -1, 3)
        
        grid = torch.bmm(self.base_grid.view(self.N, self.H * self.W, 3), theta.transpose(1, 2))
        grid = torch.div(grid[:,:,0:2],grid[:,:,2:])
        return grid


class Weighted_least_squares(nn.Module):
    def __init__(self, size, nclasses, order, cuda, reg_ls=0, use_cholesky=False, sample_factor=5):
        super().__init__()
        self.sample_factor = sample_factor
        N, C, self.H, W = size
        self.nclasses = nclasses
        self.tensor_ones = Variable(torch.ones(N,self.H*W,1))
        self.order = order
        self.reg_ls = Variable(reg_ls*torch.eye(order+1))
        self.use_cholesky = use_cholesky
        if cuda:
            self.reg_ls = self.reg_ls.cuda()
            self.tensor_ones = self.tensor_ones.cuda()
            
    def forward(self, W, grid, sample_grid):
        beta1, beta2, beta3 = None, None, None
        if sample_grid:
            W = W[:,:,::self.sample_factor,::self.sample_factor].contiguous()

        W = W.view(-1,self.nclasses, grid.size(1))
        W0 = W[:,0,:].unsqueeze(2)
        x_map = grid[:,:,0].unsqueeze(2)
        y_map = ((1)-grid[:,:,1]).unsqueeze(2)
        if self.order == 0:
            Y = self.tensor_ones
        elif self.order == 1:
            Y = torch.cat((y_map, self.tensor_ones), 2)
        elif self.order == 2:
            Y = torch.cat(((y_map**2), y_map, self.tensor_ones), 2)
        else:
            raise NotImplementedError('Requested order for polynomial fit is not implemented')
        
        Y = Y[0:W.size(0),:,:]
        x_map = x_map[0:W.size(0),:,:]
        Y0 = torch.mul(W0, Y)
        Z = torch.bmm(Y0.transpose(1,2), Y0) + self.reg_ls
        
        if not self.use_cholesky:
            Z_inv = [torch.inverse(matrix) for matrix in torch.unbind(Z)]
            Z_inv = torch.stack(Z_inv)
            X = torch.bmm(Y0.transpose(1,2), torch.mul(W0, x_map))
            beta0 = torch.bmm(Z_inv, X)
        else:
            #cholesky
            beta0=[]
            X = torch.bmm(Y0.transpose(1,2), torch.mul(W0, x_map))
            for image, rhs in zip(torch.unbind(Z),torch.unbind(X)):
                R = torch.potrf(image)
                opl = torch.trtrs(rhs, R.transpose(0,1))
                beta0.append(torch.trtrs(opl[0],R ,upper=False)[0])
            beta0 = torch.cat((beta0),1).transpose(0,1).unsqueeze(2)
        
        if self.nclasses > 1:
            W1 = W[:,1,:].unsqueeze(2)
            Y1=torch.mul(W1, Y)
            Z = torch.bmm(Y1.transpose(1,2), Y1) + self.reg_ls
            Z_inv = [torch.inverse(matrix) for matrix in torch.unbind(Z)]
            Z_inv = torch.stack(Z_inv)
            X = torch.bmm(Y1.transpose(1,2), torch.mul(W1, x_map))
            beta1 = torch.bmm(Z_inv, X)
            
        if self.nclasses > 2:
            W2 = W[:,2,:].unsqueeze(2)
            Y2=torch.mul(W2, Y)
            Z = torch.bmm(Y2.transpose(1,2), Y2) + self.reg_ls
            Z_inv = [torch.inverse(matrix) for matrix in torch.unbind(Z)]
            Z_inv = torch.stack(Z_inv)
            X = torch.bmm(Y2.transpose(1,2), torch.mul(W2, x_map))
            beta2 = torch.bmm(Z_inv, X)
            
            W3 = W[:,3,:].unsqueeze(2)
            Y3=torch.mul(W3, Y)
            Z = torch.bmm(Y3.transpose(1,2), Y3) + self.reg_ls
            Z_inv = [torch.inverse(matrix) for matrix in torch.unbind(Z)]
            Z_inv = torch.stack(Z_inv)
            X = torch.bmm(Y3.transpose(1,2), torch.mul(W3, x_map))
            beta3 = torch.bmm(Z_inv, X)
            
        return beta0, beta1, beta2, beta3
    
class Net(nn.Module):
    def __init__(self, options):
        super().__init__()
        
        #define sizes and perspective transformation
        resize = options.resize
        size = torch.Size([options.batch_size, options.nclasses, options.resize, 2*options.resize])
        size, M, M_inv = Init_Projective_transform(options.nclasses, options.batch_size, options.resize, options.grid_sample, options.sample_factor)
        self.M = Variable(M)
        
        #Init net
        self.net = define_network(options)
        #Init activation
        self.activation = activation_layer(options.activation_layer, options.cuda)
        #Init grid generator 
        self.project_layer = ProjectiveGridGenerator(size, M, options.cuda)
        #Init LS layer
        self.ls_layer = Weighted_least_squares(size, options.nclasses, options.order, options.cuda, options.reg_ls, options.use_cholesky)
        
        ############Classfication branch configuration#################
        if options.classification_branch:
            factor = options.resize/256
            if options.model_seg == 'unet' :
                chan = 512
                size_enc = (32,64)
                if options.pad == 1:
                    size_enc = tuple(int(i*factor) for i in size_enc)
                elif options.pad == 0:
                    size_enc = (16,48) if factor==1 else (24,64)
                else:
                    raise NotImplementedError
            elif options.model_seg == 'segnet':
                chan = 512
                size_enc = (8,16)
                size_enc = tuple(int(i*factor) for i in size_enc)
            elif options.model_seg == 'enet':
                chan = 128
                size_enc = (32,64)
                size_enc = tuple(int(i*factor) for i in size_enc)
            elif options.model_seg == 'resnet':
                chan = 256
                num_downsample = 0
                size_enc = tuple(int(i*factor) for i in size_enc)
                size_enc = (options.resize//(2**num_downsample),2*options.resize//(2**num_downsample))
            elif options.model_seg == 'resnet18':
                chan = 512
                size_enc = (8,16) 
                size_enc = tuple(int(i*factor) for i in size_enc)
            else:
                raise NotImplementedError
                
            self.line_classification = Classification('line', size=size_enc, channels_in=chan, resize=resize)
            self.horizon_estimation = Classification('horizon', size=size_enc, channels_in=chan, resize=resize)
        
        #DLT, proxy and classification branch initialization
        self.classification_branch = options.classification_branch
        self.DLT = DLT(options.batch_size, options.cuda, size=size_enc, channels_in=chan)
        self.proxy = Proxy_branch_segmentation(chan, resize, options.nclasses)
        self.DLT_on = options.DLT_on
        self.proxy_branch = options.proxy_branch
        
        #mask configuration
        zero_rows = ceil(options.resize*options.mask_percentage)
        self.idx_row=Variable(torch.linspace(0,zero_rows-1,zero_rows).long())
        n_row = 13
        self.idx_col1=Variable(torch.linspace(0,n_row,n_row+1).long())
        self.idx_col2=Variable(torch.linspace(0,n_row,n_row+1).long())+2*resize-(n_row+1)
        idx_mask = (np.arange(resize)[:,None] < np.arange(2*resize)-(resize+10))*1
        idx_mask = np.flip(idx_mask,1).copy() + idx_mask
        self.idx_mask = Variable(torch.from_numpy(idx_mask)).type(torch.ByteTensor).expand(options.batch_size,options.nclasses,resize,2*resize)
        
        #Place on GPU if specified
        if options.cuda:
            self.M = self.M.cuda()
            self.idx_row = self.idx_row.cuda()
            self.idx_col1 = self.idx_col1.cuda()
            self.idx_col2 = self.idx_col2.cuda()
            self.idx_mask = self.idx_mask.cuda()
            if options.classification_branch:
                self.line_classification = self.line_classification.cuda()
                self.horizon_estimation = self.horizon_estimation.cuda()
                self.DLT = self.DLT.cuda()
            if options.proxy_branch:
                self.proxy = self.proxy.cuda()
                
    def forward(self, input, no_ortho_view, sample_grid):
        output_seg, line, horizon, x = None, None, None, None
        #1. Push bach trough network
        output, output_encoder = self.net(input)
        
        #2. Classfication branch for line and horizon estimation
        if self.classification_branch:
            line = self.line_classification(output_encoder)
            horizon = self.horizon_estimation(output_encoder)
            
        #3. Proxy branch for segmentation
        if self.proxy_branch:
            output_seg = self.proxy(output_encoder)
        
        #4. Use activation function 
        output = self.activation(output)
            
        #5. Use mask 
        if not no_ortho_view:
            output = output.index_fill(2,self.idx_row,0)
#            output = output.index_fill(3,self.idx_col1,0)
#            output = output.index_fill(3,self.idx_col2,0)
#            output = output.masked_fill(self.idx_mask,0)

        #6. DLT transform
        if self.DLT_on:
            M, x = self.DLT(output_encoder)
            grid = self.project_layer(M, no_ortho_view)
        else:
            M = self.M
            grid = self.project_layer(self.M, no_ortho_view)
        
        #7. LS layer
        beta0, beta1, beta2, beta3 = self.ls_layer(output, grid, sample_grid)
        return beta0, beta1, beta2, beta3, output**2, output_seg, line, horizon, M, x


def define_network(options, norm='batch'):
    print('defining network')
    if options.model_seg == 'resnet':
        net = simple_net(options.nclasses)
    elif options.model_seg == 'resnet18':
        net = ResNet18(options.nclasses)
    elif options.model_seg == 'unet':
        net = UNet(options.nclasses, options.activation_net, options.pad)
    elif options.model_seg == 'enet':
        net = ENet(options.nclasses, options.norm, options.no_dropout)
    elif options.model_seg == 'segnet':
        net = SegNet(options.nclasses)
    else:
        raise NotImplementedError('The requested {} is not yet implemented'.format(options.model_seg))
    if options.cuda:
        net = net.cuda()
    print(net)
    return net

def define_optimizer(params, options):
    print('define optimizer')
    if options.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=options.learning_rate, weight_decay=options.weight_decay)
    elif options.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=options.learning_rate, momentum=0.9, weight_decay=options.weight_decay)
    else:
        raise NotImplementedError
    return optimizer

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.001, patience=5)
    elif opt.lr_policy == 'none':
        scheduler = None
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def square_tensor(x):
    return x**2

def return_tensor(x):
    return x

def activation_layer(activation='square', cuda=True):
    place_cuda = True
    if activation == 'sigmoid':
        layer = nn.Sigmoid()
    elif activation == 'relu':
        layer = nn.ReLU()
    elif activation == 'softplus':
        layer = nn.Softplus()
    elif activation =='square':
        layer = square_tensor
        place_cuda = False
    elif activation == 'abs':
        layer = torch.abs
        place_cuda = False
    elif activation == 'none':
        layer = return_tensor
    else:
        raise NotImplementedError('Activation type: {} is not implemented'.format(activation))
    if cuda and place_cuda:
        layer = layer.cuda()
    return layer

def init_weights(net, init_w='normal', activation='relu'):
    print('Init weights in network with [{}]'.format(init_w))
    if init_w == 'normal':
        net.apply(weights_init_normal)
    elif init_w == 'xavier':
        net.apply(weights_init_xavier)
    elif init_w == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_w == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{}] is not implemented'.format(init))

def weights_init_normal(m):
    classname = m.__class__.__name__
#    print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
        

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
#    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

