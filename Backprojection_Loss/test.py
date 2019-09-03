"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import numpy as np
import torch
import torch.optim
import torch.nn as nn
from eval_lane import LaneEval
from tqdm import tqdm
from Networks.utils import get_homography
import os
import json
import cv2
from PIL import Image
from Networks.utils import AverageMeter
import time

def resize_coordinates(array):
    return array*2.5

def test_model(loader, model, criterion, criterion_seg, 
               criterion_line_class, criterion_horizon, args, epoch=0):
    # Init 
    assert args.end_to_end == True
    params = Projections(args)
    gt_file = os.path.join(args.test_dir, 'test_label.json')
    gt_lanes = [json.loads(line) for line in open(gt_file,'r')]
    test_set_file = os.path.join(args.save_path, 'test_set_predictions.json')
    colormap = [(255,0,0), (0,255,0), (255,255,0), (0,0,255), (0, 128, 128)]
    batch_time = AverageMeter()

    # Evaluate model
    model.eval()

    # Only forward pass, hence no gradients needed
    with torch.no_grad():
        with open(test_set_file, 'w') as jsonFile:
            # Start validation loop
            for i, input in tqdm(enumerate(loader)):

                # Reset coordinates
                x_cal0, x_cal1, x_cal2, x_cal3 = [None]*4

                # Put inputs on gpu if possible
                if not args.no_cuda:
                    input = input.cuda(non_blocking=True).float()

                # Run model
                torch.cuda.synchronize()
                a = time.time()
                beta0, beta1, beta2, beta3, weightmap_zeros, \
                    output_net, outputs_line, outputs_horizon, output_seg = model(input, gt_line=np.array([1,1]), 
                                                                                  end_to_end=args.end_to_end, gt=None)
                torch.cuda.synchronize()
                b = time.time()
                batch_time.update(b-a)

                # Horizon task & Line classification task
                if args.clas:
                    horizon_pred = nn.Sigmoid()(outputs_horizon).sum(dim=1)
                    horizon_pred = (torch.round((resize_coordinates(horizon_pred) + 80)/10)*10).int()
                    line_pred = torch.round(nn.Sigmoid()(outputs_line))
                else:
                    assert False

                # Calculate X coordinates
                x_cal0 = params.compute_coordinates(beta0)
                x_cal1 = params.compute_coordinates(beta1)
                x_cal2 = params.compute_coordinates(beta2)
                x_cal3 = params.compute_coordinates(beta3)
                lanes_pred = torch.stack((x_cal0, x_cal1, x_cal2, x_cal3), dim=1)

                # Check line type branch
                line_pred = line_pred[:, [1, 2, 0, 3]]
                lanes_pred[(1 - line_pred[:, :, None]).byte().expand_as(lanes_pred)] = -2

                # Check horizon branch
                bounds = ((horizon_pred - 160) / 10)
                for k, bound in enumerate(bounds):
                    lanes_pred[k, :, :bound.item()] = -2

                # TODO check intersections
                lanes_pred[lanes_pred > 1279] = -2
                lanes_pred[lanes_pred < 0] = -2

                # Write predictions to json file
                lanes_pred = np.int_(np.round(lanes_pred.data.cpu().numpy())).tolist()
                num_el = input.size(0)

                for j in range(num_el):
                    lanes_to_write = lanes_pred[j]
                    im_id = i*args.val_batch_size + j 
                    json_line = gt_lanes[im_id]
                    json_line["lanes"] = lanes_to_write
                    json_line["run_time"] = 20
                    json.dump(json_line, jsonFile)
                    jsonFile.write('\n')

                    if args.draw_testset:
                        test = weightmap_zeros[j]
                        weight0= test[0]
                        weight1= test[1]
                        weight2= test[2]
                        weight3= test[3]
                        to_vis = weight0/weight0.max()+weight1/weight1.max()+weight2/weight2.max()+weight3/weight3.max()
                        to_vis = to_vis.data.cpu().numpy()
                        to_vis = Image.fromarray(to_vis)
                        to_vis.save(os.path.join(args.save_path + '/example/testset', '{}_vis.jpg'.format(im_id))) 
                        im_path = json_line['raw_file']
                        img_name = os.path.join(args.test_dir, im_path)
                        with open(img_name, 'rb') as f:
                            img = np.array(Image.open(f).convert('RGB'))
                        for lane_i in range(len(lanes_to_write)):
                            x_orig = lanes_to_write[lane_i]
                            pt_or = [(xcord, ycord) for (xcord, ycord) in zip(x_orig, json_line['h_samples']) if xcord!=-2]
                            for point in pt_or:
                               img = cv2.circle(img, tuple(np.int32(point)), thickness=-1, color=colormap[lane_i], radius = 3)
                        img = Image.fromarray(np.uint8(img))
                        img.save(os.path.join(args.save_path + '/example/testset', '{}.jpg'.format(im_id))) 


        # Calculate accuracy
        if args.clas and args.nclasses > 3:
            acc_seg = LaneEval.bench_one_submit(test_set_file, gt_file)
            print(acc_seg)
            print("===> Average ACC on TESTSET is {:.8} in {:.6}s for a batch".format(acc_seg[0], batch_time.avg))
    return acc_seg[0]


class Projections():
    '''
    Compute coordintes after backprojection to original perspective
    '''
    def __init__(self, options):
        super(Projections, self).__init__()
        M, M_inv = get_homography(resize=options.resize, no_mapping=False)
        self.M, self.M_inv = torch.from_numpy(M), torch.from_numpy(M_inv)
        start = 160
        delta = 10
        num_heights = (720-start)//delta
        self.y_d = (torch.arange(start,720,delta)-80).double() / 2.5
        self.ones = torch.ones(num_heights).double()
        self.y_prime = (self.M[1,1:2]*self.y_d + self.M[1,2:])/(self.M[2,1:2]*self.y_d+self.M[2,2:])
        self.y_eval = 255 - self.y_prime
        self.Y = torch.stack((self.y_eval**2, self.y_eval, self.ones), 1)

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

        # use gpu
        self.M = self.M.cuda()
        self.M_inv = self.M_inv.cuda()
        self.y_prime = self.y_prime.cuda()
        self.Y = self.Y.cuda()
        self.ones = self.ones.cuda()

    def compute_coordinates(self, params):
        # Sample at y_d in the homography space
        bs = params.size(0)
        x_prime = torch.bmm(self.Y[:bs], params)

        # Transform sampled points back
        coordinates = torch.stack((x_prime, self.y_prime[:bs], self.ones[:bs]), 2).squeeze(3).permute((0, 2, 1))
        trans = torch.bmm(self.M_inv[:bs], coordinates)
        x_cal = trans[:,0,:]/trans[:,2,:]
        # y_cal = trans[:,1,:]/trans[:,2,:] # sanity check

        # Rezize
        x_cal = resize_coordinates(x_cal)

        return x_cal
