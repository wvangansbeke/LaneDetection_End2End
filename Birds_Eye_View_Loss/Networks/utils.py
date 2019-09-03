"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import cv2
import argparse
import numpy as np
import os
import sys
import errno
from PIL import Image
import torch
import torch.optim
from torch.optim import lr_scheduler
import torch.nn.init as init
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
plt.rcParams['figure.figsize'] = (35, 30)


def define_args():
    parser = argparse.ArgumentParser(description='Lane_detection_all_objectives')
    # Segmentation model settings
    parser.add_argument('--dataset', default='lane_detection', help='dataset images to train on')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--nepochs', type=int, default=350, help='total numbers of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--no_cuda', action='store_true', help='if gpu available')
    parser.add_argument('--nworkers', type=int, default=8, help='num of threads')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout in network')
    parser.add_argument('--nclasses', type=int, default=2, choices=[2, 4], help='num output channels for segmentation')
    parser.add_argument('--crop_size', type=int, default=80, help='crop from image')
    parser.add_argument('--resize', type=int, default=256, help='resize image to resize x (ratio*resize)')
    parser.add_argument('--mod', type=str, default='erfnet', help='model to train')
    parser.add_argument('--layers', type=int, default=18, help='amount of layers in model')
    parser.add_argument("--pool", type=str2bool, nargs='?', const=True, default=True, help="use pooling")
    parser.add_argument("--pretrained", type=str2bool, nargs='?', const=True, default=False, help="use pretrained model")
    parser.add_argument('--pretrain_epochs', type=int, default=20, help='Number of epochs to perform segmentation pretraining')
    parser.add_argument('--channels_in', type=int, default=3, help='num channels of input image')
    parser.add_argument('--norm', type=str, default='batch', help='normalisation layer you want to use')
    parser.add_argument('--flip_on', action='store_true', help='Random flip input images on?')
    parser.add_argument('--num_train', type=int, default=2535, help='Train on how many images of trainset')
    parser.add_argument('--split_percentage', type=float, default=0.2, help='where to split dataset in train and validationset')
    parser.add_argument('--test_mode', action='store_true', help='prevents loading latest saved model')
    parser.add_argument('--start_epoch', type=int, default=0, help='prevents loading latest saved model')
    parser.add_argument('--evaluate', action='store_true', help='only perform evaluation')
    parser.add_argument('--resume', type=str, default='', help='resume latest saved run')
    # Optimizer settings
    parser.add_argument('--optimizer', type=str, default='adam', help='adam or sgd')
    parser.add_argument('--weight_init', type=str, default='kaiming', help='normal, xavier, kaiming, orhtogonal weights initialisation')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 weight decay/regularisation on?')
    parser.add_argument('--lr_decay', action='store_true', help='decay learning rate with rule')
    parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=400, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--lr_policy', default=None, help='learning rate policy: lambda|step|plateau')
    parser.add_argument('--lr_decay_iters', type=int, default=30, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--clip_grad_norm', type=int, default=0, help='performs gradient clipping')
    # Fitting layer settings
    parser.add_argument('--order', type=int, default=2, help='order of polynomial for curve fitting')
    parser.add_argument('--activation_layer', type=str, default='square', help='Which activation after decoder do you want?')
    parser.add_argument('--reg_ls', type=float, default=0, help='Regularization term for matrix inverse')
    parser.add_argument('--no_ortho', action='store_true', help='if no ortho transformation is desired')
    parser.add_argument('--mask_percentage', type=float, default=0.3, help='mask to apply where birds eye view is not defined')
    parser.add_argument('--use_cholesky', action='store_true', help='use cholesky decomposition')
    parser.add_argument('--activation_net', type=str, default='relu', help='activation in network used')
    # Paths settings
    parser.add_argument('--image_dir', type=str, required=True, help='directory to image dir')
    parser.add_argument('--gt_dir', type=str, required=True, help='directory to gt')
    parser.add_argument('--save_path', type=str, default='Saved/', help='directory to gt')
    parser.add_argument('--json_file', type=str, default='Labels/Curve_parameters.json', help='directory to json input')
    # LOSS settings
    parser.add_argument('--weight_seg', type=int, default=30, help='weight in loss criterium for segmentation')
    parser.add_argument('--weight_class', type=float, default=1, help='weight in loss criterium for classification branch')
    parser.add_argument('--weight_fit', type=float, default=1, help='weight in loss criterium for fit')
    parser.add_argument('--loss_policy', type=str, default='area', help='use area_loss, homography_mse or classical mse in birds eye view')
    parser.add_argument('--weight_funct', type=str, default='none', help='apply weight function in birds eye when computing area loss')
    parser.add_argument("--end_to_end", type=str2bool, nargs='?', const=True, default=True, help="regression towards curve params by network or postprocessing")
    parser.add_argument('--gamma', type=float, default=0., help='factor to decay learning rate every lr_decay_iters with')
    parser.add_argument("--clas", type=str2bool, nargs='?', const=True, default=False, help="Horizon and line classification tasks")
    # CUDNN usage
    parser.add_argument("--cudnn", type=str2bool, nargs='?', const=True, default=True, help="cudnn optimization active")
    # Tensorboard settings
    parser.add_argument("--no_tb", type=str2bool, nargs='?', const=True, default=True, help="Use tensorboard logging by tensorflow")
    # Print settings
    parser.add_argument('--print_freq', type=int, default=500, help='padding')
    parser.add_argument('--save_freq', type=int, default=500, help='padding')
    # Skip batch
    parser.add_argument('--list', type=int, nargs='+', default=[954, 2789], help='Images you want to skip')
    return parser


def save_weightmap(train_or_val, M, M_inv, weightmap_zeros,
                   beta0, beta1, beta2, beta3, gt_params_lhs, 
                   gt_params_rhs, gt_params_llhs, gt_params_rrhs, line_class,
                   gt, idx, i, images, no_ortho, resize, save_path):
    M = M.data.cpu().numpy()[0]
    x = np.zeros(3)

    line_class = line_class[0].cpu().numpy()
    left_lane = True if line_class[0] != 0 else False
    right_lane = True if line_class[3] != 0 else False

    wm0_zeros = weightmap_zeros.data.cpu()[0, 0].numpy()
    wm1_zeros = weightmap_zeros.data.cpu()[0, 1].numpy()

    im = images.permute(0, 2, 3, 1).data.cpu().numpy()[0]
    im_orig = np.copy(im)
    gt_orig = gt.permute(0, 2, 3, 1).data.cpu().numpy()[0, :, :, 0]
    im_orig = draw_homography_points(im_orig, x, resize)

    im, M_scaledup = test_projective_transform(im, resize, M)

    im, _ = draw_fitted_line(im, gt_params_rhs[0], resize, (0, 255, 0))
    im, _ = draw_fitted_line(im, gt_params_lhs[0], resize, (0, 255, 0))
    im, lane0 = draw_fitted_line(im, beta0[0], resize, (255, 0, 0))
    im, lane1 = draw_fitted_line(im, beta1[0], resize, (0, 0, 255))
    if beta2 is not None:
        im, _ = draw_fitted_line(im, gt_params_llhs[0], resize, (0, 255, 0))
        im, _ = draw_fitted_line(im, gt_params_rrhs[0], resize, (0, 255, 0))
        if left_lane:
            im, lane2 = draw_fitted_line(im, beta2[0], resize, (255, 255, 0))
        if right_lane:
            im, lane3 = draw_fitted_line(im, beta3[0], resize, (255, 128, 0))


    if not no_ortho:
        im_inverse = cv2.warpPerspective(im, np.linalg.inv(M_scaledup), (2*resize, resize))
    else:
        im_inverse = im_orig

    im_orig = np.clip(im_orig, 0, 1)
    im_inverse = np.clip(im_inverse, 0, 1)
    im = np.clip(im, 0, 1)

    fig = plt.figure()
    ax1 = fig.add_subplot(421)
    ax2 = fig.add_subplot(422)
    ax3 = fig.add_subplot(423)
    ax4 = fig.add_subplot(424)
    ax5 = fig.add_subplot(425)
    ax6 = fig.add_subplot(426)
    ax7 = fig.add_subplot(427)
    ax1.imshow(im)
    ax2.imshow(wm0_zeros)
    ax3.imshow(im_inverse)
    ax4.imshow(wm1_zeros)
    ax5.imshow(wm0_zeros/np.max(wm0_zeros)+wm1_zeros/np.max(wm1_zeros))
    ax6.imshow(im_orig)
    ax7.imshow(gt_orig)
    fig.savefig(save_path + '/example/{}/weight_idx-{}_batch-{}'.format(train_or_val, idx, i))
    plt.clf()
    plt.close(fig)


def test_projective_transform(input, resize, M):
    # test grid using built in F.grid_sampler method.
    M_scaledup = np.array([[M[0,0],M[0,1]*2,M[0,2]*(2*resize-1)],[0,M[1,1],M[1,2]*(resize-1)],[0,M[2,1]/(resize-1),M[2,2]]])
    inp = cv2.warpPerspective(np.asarray(input), M_scaledup, (2*resize,resize))
    return inp, M_scaledup


def draw_fitted_line(img, params, resize, color=(255,0,0)):
    params = params.data.cpu().tolist()
    y_stop = 0.7
    y_prime = np.linspace(0, y_stop, 20)
    params = [0] * (4 - len(params)) + params
    d, a, b, c = [*params]
    x_pred = d*(y_prime**3) + a*(y_prime)**2 + b*(y_prime) + c
    x_pred = x_pred*(2*resize-1)
    y_prime = (1-y_prime)*(resize-1)
    lane = [(xcord, ycord) for (xcord, ycord) in zip(x_pred, y_prime)] 
    img = cv2.polylines(img, [np.int32(lane)], isClosed = False, color = color,thickness = 1)
    return img, lane


def draw_horizon(img, horizon, resize=256, color=(255,0,0)):
    x = np.arange(2*resize-1)
    horizon_line = [(x_cord, horizon+1) for x_cord in x]
    img = cv2.polylines(img.copy(), [np.int32(horizon_line)], isClosed = False, color = color,thickness = 1)
    return img


def draw_homography_points(img, x, resize=256, color=(255,0,0)):
    y_start1 = (0.3+x[2])*(resize-1)
    y_start = 0.3*(resize-1)
    y_stop = resize-1
    src = np.float32([[0.45*(2*resize-1),y_start],[0.55*(2*resize-1), y_start],[0.1*(2*resize-1),y_stop],[0.9*(2*resize-1), y_stop]])
    dst = np.float32([[(0.45+x[0])*(2*resize-1), y_start1],[(0.55+x[1])*(2*resize-1), y_start1],[(0.45+x[0])*(2*resize-1), y_stop],[(0.55+x[1])*(2*resize-1),y_stop]])
    dst_ideal = np.float32([[0.45*(2*resize-1), y_start],[0.55*(2*resize-1), y_start],[0.45*(2*resize-1), y_stop],[0.55*(2*resize-1),y_stop]])
    [cv2.circle(np.asarray(img), tuple(idx), radius=5, thickness=-1, color=(255,0,0)) for idx in src]
    [cv2.circle(np.asarray(img), tuple(idx), radius=5, thickness=-1, color=(0,255,0)) for idx in dst_ideal]
    [cv2.circle(np.asarray(img), tuple(idx), radius=5, thickness=-1, color=(0,0,255)) for idx in dst]
    return img


def save_image(output, gt_params, i=1, resize=320):
    outputs_seg = output.permute(0,2,3,1)
    im = np.asarray(outputs_seg.data.cpu()[0])
    im = draw_fitted_line(im,gt_params[0],resize)
    im = Image.fromarray(im.astype('uint8'), 'RGB')
    im.save('simple_net/simple_net_train/{}.png'.format(i[0]))


def save_output(output,gt_params, i=1):
    output = output*255/(torch.max(output))
    output = output.permute(0,2,3,1)
    im = np.asarray(output.data.cpu()[0]).squeeze(2)
    
#    im = draw_fitted_line(im,gt_params[0],resize)
    im = Image.fromarray(im.astype('uint8'), 'P')
    im.save('simple_net/simple_net_output/{}.png'.format(i[0]))


def line_right_eq(x):
    y = 0.438/0.7*x + 0.56 #0.7/0.438*(x-0.56)
    return y


def line_left_eq(x):
    y = -x*0.438/0.7 + 0.44#-0.7/0.438*(x-0.44)
    return y


def f(x, *params):
    '''
    Constructs objective function which will be solved iteratively
    '''
    a, b, c, left_or_right = params
    if left_or_right == 'left':
        funct = a*x**2 + b*x + c - line_left_eq(x)
    else:
        funct = a*x**2 + b*x + c - line_right_eq(x)
    return funct


def draw_mask_line(im, beta0, beta1, beta2, beta3, resize):
    beta0 = beta0.data.cpu().tolist()
    beta1 = beta1.data.cpu().tolist()
    beta2 = beta2.data.cpu().tolist()
    beta3 = beta3.data.cpu().tolist()
    params0 = *beta0, 'left'
    params1 = *beta1, 'right'
    params2 = *beta2, 'left'
    params3 = *beta3, 'right'
    x, y, order = [], [], []
    max_lhs = fsolve(f, 0.05, args=params0)
    if max_lhs > 0:
        x.append(line_left_eq(max_lhs[0]))
        y.append(1-max_lhs[0])
        order.append(0)
    else:
        max_lhs = 0
    max_rhs = fsolve(f, 0.05, args=params1)
    if max_rhs > 0:
        x.append(line_right_eq(max_rhs[0]))
        y.append(1-max_rhs[0])
        order.append(1)
    else:
        max_rhs = 0 
    max_left = fsolve(f, 0.05, args=params2)
    if max_left > 0:
        x.append(line_left_eq(max_left[0]))
        y.append(1-max_left[0])
        order.append(2)
    else:
        max_left = 0
    max_right = fsolve(f, 0.05, args=params3)
    if max_right > 0:
        x.append(line_right_eq(max_right[0]))
        y.append(1-max_right[0])
        order.append(3)
    else:
        max_right = 0 
    y_stop = 1
    y_prime = np.linspace(0, y_stop, 40)
    x_prime_right = line_right_eq(y_prime)
    x_prime_left = line_left_eq(y_prime)
    y_prime, x_prime_lft, x_prime_rght = (1-y_prime)*(resize-1), x_prime_left*(2*resize-1), x_prime_right*(2*resize-1)
    line_right = [(xcord, ycord) for (xcord, ycord) in zip(x_prime_rght, y_prime)] 
    line_left = [(xcord, ycord) for (xcord, ycord) in zip(x_prime_lft, y_prime)] 
    im = cv2.polylines(im, [np.int32(line_right)], isClosed = False, color = (255,0,0),thickness = 1)
    im = cv2.polylines(im, [np.int32(line_left)], isClosed = False, color = (255,0,0),thickness = 1)
    x = np.array(x)
    y = np.array(y)
    x_left, x_right = line_left_eq(max_left)*(2*resize-1), line_right_eq(max_right)*(2*resize-1)
    y_left, y_right = (1-max_left)*(resize-1), (1-max_right)*(resize-1)
    cv2.circle(np.asarray(im), (x_left,y_left), radius=3, thickness=-1, color=(0,0,255))
    cv2.circle(np.asarray(im), (x_right,y_right), radius=3, thickness=-1, color=(0,0,255))
    x_prime, y_prime = homogenous_transformation(x,y)
    maxima = np.zeros(4)
    for i, idx in enumerate(order):
        maxima[idx] = y_prime[i]
    return im, np.int_(np.round(x_prime*(2*resize-1))), np.int_(np.round(y_prime*(resize-1)))


def homogenous_transformation(x,y):
    """
    Helper function to transform coordionates defined by transformation matrix
    
    Args:
            Matrix (multi dim - array): Transformation matrix
            x (array): original x coordinates
            y (array): original y coordinates
    """
    y_start = 0.3
    y_stop = 1
    src = np.float32([[0.45,y_start],[0.55, y_start],[0.1,y_stop],[0.9, y_stop]])
    dst = np.float32([[0.45, y_start],[0.55, y_start],[0.45, y_stop],[0.55,y_stop]])
    M_inv = cv2.getPerspectiveTransform(dst,src)
    
    ones = np.ones((1,len(y)))
    coordinates = np.vstack((x, y, ones))
    trans = np.matmul(M_inv, coordinates)
            
    x_vals = trans[0,:]/trans[2,:]
    y_vals = trans[1,:]/trans[2,:]
    return x_vals, y_vals


def first_run(save_path):
    txt_file = os.path.join(save_path,'first_run.txt')
    if not os.path.exists(txt_file):
        open(txt_file, 'w').close()
    else:
        saved_epoch = open(txt_file).read()
        if saved_epoch is None:
            print('You forgot to delete [first run file]')
            return '' 
        return saved_epoch
    return ''


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


# trick from stackoverflow
def str2bool(argument):
    if argument.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif argument.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Wrong argument in argparse, should be a boolean')


class Logger(object):
    """
    Source https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        self.fpath = fpath
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def define_optim(optim, params, lr, weight_decay):
    if optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("The requested optimizer: {} is not implemented".format(optim))
    return optimizer


def define_scheduler(optimizer, args):
    if args.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - args.niter) / float(args.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=args.lr_decay_iters, gamma=args.gamma)
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                   factor=args.gamma,
                                                   threshold=0.0001,
                                                   patience=args.lr_decay_iters)
    elif args.lr_policy == 'none':
        scheduler = None
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def define_init_weights(model, init_w='normal', activation='relu'):
    print('Init weights in network with [{}]'.format(init_w))
    if init_w == 'normal':
        model.apply(weights_init_normal)
    elif init_w == 'xavier':
        model.apply(weights_init_xavier)
    elif init_w == 'kaiming':
        model.apply(weights_init_kaiming)
    elif init_w == 'orthogonal':
        model.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{}] is not implemented'.format(init_w))


def weights_init_normal(m):
    classname = m.__class__.__name__
#    print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
#    print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.orthogonal(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
