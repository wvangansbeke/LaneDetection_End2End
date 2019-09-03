"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from Networks.utils import get_homography

class polynomial():
    """
    Polynomial class with exact integral calculation according to the
    trapezium rule.
    """
    def __init__(self, coeffs, a=0, b=0.7, n=100):
        self.a1, self.b1, self.c1 = torch.chunk(coeffs, 3, 1)
        self.a1, self.b1, self.c1 = self.a1.squeeze(), self.b1.squeeze(), self.c1.squeeze()
        self.a = a
        self.b = b
        self.n = n

    def calc_pol(self, x):
        return self.a1*x**2 + self.b1*x + self.c1

    def trapezoidal(self, other):
        h = float(self.b - self.a) / self.n
        s = 0.0
        s += abs(self.calc_pol(self.a)/2.0 - other.calc_pol(other.a)/2.0)
        for i in range(1, self.n):
            s += abs(self.calc_pol(self.a + i*h) - other.calc_pol(self.a + i*h))
        s += abs(self.calc_pol(self.b)/2.0 - other.calc_pol(self.b)/2.0)
        out = s*h
        return out


# pol1 = polynomial(coeffs=torch.FloatTensor([[0, 1, 0]]), a=-1, b=1)
# pol2 = polynomial(coeffs=torch.FloatTensor([[0, 0, 0]]), a=-1, b=1)
# pol1 = polynomial(coeffs=torch.FloatTensor([[0, 1, 0]]), a=0, b=1)
# pol2 = polynomial(coeffs=torch.FloatTensor([[1, 0, 0]]), a=0, b=1)
# print('Area by trapezium rule is {}'.format(pol1.trapezoidal(pol2)))


def define_loss_crit(options):
    '''
    Define loss cirterium:
        -MSE loss on curve parameters in ortho view
        -MSE loss on points after backprojection to normal view
        -Area loss
    '''
    if options.loss_policy == 'mse':
        loss_crit = MSE_Loss(options)
    elif options.loss_policy == 'homography_mse':
        loss_crit = Homography_MSE_Loss(options)
    elif options.loss_policy == 'backproject':
        loss_crit = backprojection_loss(options)
    elif options.loss_policy == 'area':
        loss_crit = Area_Loss(options.order, options.weight_funct)
    else:
        return NotImplementedError('The requested loss criterion is not implemented')
    weights = (torch.Tensor([1] + [options.weight_seg]*(options.nclasses))).cuda()
    loss_seg = nn.CrossEntropyLoss(weights)
    # loss_seg = CrossEntropyLoss2d(seg=True, nclasses=options.nclasses, weight=options.weight_seg)
    return loss_crit, loss_seg


class CrossEntropyLoss2d(nn.Module):
    '''
    Standard 2d cross entropy loss on all pixels of image
    My implemetation (but since Pytorch 0.2.0 libs have their
    owm optimized implementation, consider using theirs)
    '''
    def __init__(self, weight=None, size_average=True, seg=False, nclasses=2):
        if seg:
            weights = torch.Tensor([1] + [weight]*(nclasses))
            weights = weights.cuda()
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weights, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets[:, 0, :, :])


class Area_Loss(nn.Module):
    '''
    Compute area between curves by integrating (x1 - x2)^2 over y
    *Area:
        *order 0: int((c1 - c2)**2)dy
        *order 1: int((b1*y - b2*y + c1 - c2)**2)dy
        *order 2: int((a1*y**2 - a2*y**2 + b1*y - b2*y + c1 - c2)**2)dy

    *A weight function W can be added:
        Weighted area: int(W(y)*diff**2)dy
        with W(y):
            *1
            *(1-y)
            *(1-y**0.5)
    '''
    def __init__(self, order, weight_funct):
        super(Area_Loss, self).__init__()
        self.order = order
        self.weight_funct = weight_funct

    def forward(self, params, gt_params, compute=True):
        diff = params.squeeze(-1) - gt_params
        a = diff[:, 0]
        b = diff[:, 1]
        t = 0.7 # up to which y location to integrate
        if self.order == 2:
            c = diff[:, 2]
            if self.weight_funct == 'none':
                # weight (1)
                loss_fit = (a**2)*(t**5)/5+2*a*b*(t**4)/4 + \
                           (b**2+c*2*a)*(t**3)/3+2*b*c*(t**2)/2+(c**2)*t
            elif self.weight_funct == 'linear':
                # weight (1-y)
                loss_fit = c**2*t - t**5*((2*a*b)/5 - a**2/5) + \
                           t**2*(b*c - c**2/2) - (a**2*t**6)/6 - \
                           t**4*(b**2/4 - (a*b)/2 + (a*c)/2) + \
                           t**3*(b**2/3 - (2*c*b)/3 + (2*a*c)/3)
            elif self.weight_funct == 'quadratic':
                # weight (1-y**0.5)
                loss_fit = t**3*(1/3*b**2 + 2/3*a*c) - \
                           t**(7/2)*(2/7*b**2 + 4/7*a*c) + \
                           c**2*t + 0.2*a**2*t**5 - 2/11*a**2*t**(11/2) - \
                           2/3*c**2*t**(3/2) + 0.5*a*b*t**4 - \
                           4/9*a*b*t**(9/2) + b*c*t**2 - 0.8*b*c*t**(5/2)
            else:
                return NotImplementedError('The requested weight function is \
                        not implemented, only order 1 or order 2 possible')
        elif self.order == 1:
            loss_fit = (b**2)*t + a*b*(t**2) + ((a**2)*(t**3))/3
        else:
            return NotImplementedError('The requested order is not implemented, only none, linear or quadratic possible')

        # Mask select if lane is present
        mask = torch.prod(gt_params != 0, 1).byte()
        loss_fit = torch.masked_select(loss_fit, mask)
        loss_fit = loss_fit.mean(0) if loss_fit.size()[0] != 0 else 0 # mean over the batch
        return loss_fit


class MSE_Loss(nn.Module):
    '''
    Compute mean square error loss on curve parameters
    in ortho or normal view
    '''
    def __init__(self, options):
        super(MSE_Loss, self).__init__()
        self.loss_crit = nn.MSELoss()
        if not options.no_cuda:
            self.loss_crit = self.loss_crit.cuda()

    def forward(self, params, gt_params, compute=True):
        loss = self.loss_crit(params.squeeze(-1), gt_params)
        return loss

class backprojection_loss(nn.Module):
    '''
    Compute mean square error loss on points in normal view
    instead of parameters in ortho view
    '''
    def __init__(self, options):
        super(backprojection_loss, self).__init__()
        M, M_inv = get_homography(options.resize, options.no_mapping)
        self.M, self.M_inv = torch.from_numpy(M), torch.from_numpy(M_inv)
        start = 160
        delta = 10
        num_heights = (720-start)//delta
        self.y_d = (torch.arange(start,720,delta)-80).double() / 2.5
        self.ones = torch.ones(num_heights).double()
        self.y_prime = (self.M[1,1:2]*self.y_d + self.M[1,2:])/(self.M[2,1:2]*self.y_d+self.M[2,2:])
        self.y_eval = 255 - self.y_prime

        if options.order == 0:
            self.Y = self.tensor_ones
        elif options.order == 1:
            self.Y = torch.stack((self.y_eval, self.ones), 1)
        elif options.order == 2:
            self.Y = torch.stack((self.y_eval**2, self.y_eval, self.ones), 1)
        elif options.order == 3:
            self.Y = torch.stack((self.y_eval**3, self.y_eval**2, self.y_eval, self.ones), 1)
        else:
            raise NotImplementedError(
                    'Requested order {} for polynomial fit is not implemented'.format(options.order))

        self.Y = self.Y.unsqueeze(0).repeat(options.batch_size, 1, 1)
        self.ones = torch.ones(options.batch_size, num_heights, 1).double()
        self.y_prime = self.y_prime.unsqueeze(0).repeat(options.batch_size, 1).unsqueeze(2)
        self.M_inv = self.M_inv.unsqueeze(0).repeat(options.batch_size, 1, 1)

        if not options.no_cuda:
            self.M = self.M.cuda()
            self.M_inv = self.M_inv.cuda()
            self.y_prime = self.y_prime.cuda()
            self.Y = self.Y.cuda()
            self.ones = self.ones.cuda()

    def forward(self, params, x_gt, valid_samples):
        # Sample at y_d in the homography space
        bs = params.size(0)
        x_prime = torch.bmm(self.Y[:bs], params)

        # Transform sampled points back
        coordinates = torch.stack((x_prime, self.y_prime[:bs], self.ones[:bs]), 2).squeeze(3).permute((0, 2, 1))
        trans = torch.bmm(self.M_inv[:bs], coordinates)
        x_cal = trans[:,0,:]/trans[:,2,:]
        # y_cal = trans[:,1,:]/trans[:,2,:] # sanity check

        # Compute error
        x_err = (x_gt-x_cal)*valid_samples
        loss = torch.sum(x_err**2) / (valid_samples.sum())
        if valid_samples.sum() == 0:
            loss = 0
        return loss, x_cal * valid_samples
