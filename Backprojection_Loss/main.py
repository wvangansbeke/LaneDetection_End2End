"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import numpy as np
import torch
import torch.optim
import torch.nn as nn

import os
import glob
import time
import sys
import shutil
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from Dataloader.Load_Data_new import get_loader, get_testloader, \
                                     load_valid_set_file_all 
                                    
# from eval_lane import LaneEval
from Loss_crit import define_loss_crit, polynomial 
from Networks.LSQ_layer import Net
from Networks.utils import define_args, save_weightmap, first_run, \
                           mkdir_if_missing, Logger, define_init_weights,\
                           define_scheduler, define_optim, AverageMeter
from test import test_model


def get_flags(epoch=0):
    # Set flags for pretraining
    skip = False
    if args.pretrained:
        if (epoch < args.pretrain_epochs):
            args.end_to_end = False
            print("Pretraining so set args.end_to_end to {}".format(args.end_to_end))
            if epoch < args.skip_epochs:
                print("Skipping LSQ layer")
                skip=True
        else:
            args.end_to_end = True
    return skip

def main():
    global args
    global mse_policy
    parser = define_args()
    args = parser.parse_args()
    if not args.end_to_end:
        assert args.pretrained == False
    if args.clas:
        assert args.nclasses == 4
    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size
    

    # Check GPU availability
    if not args.no_cuda and not torch.cuda.is_available():
        raise Exception("No gpu available for usage")
    torch.backends.cudnn.benchmark = args.cudnn

    # Define save path
    save_id = 'Mod_{}_opt_{}_loss_{}_lr_{}_batch_{}_end2end_{}_chol_{}_lanes_{}_pretrain{}_clas{}_mask{}_flip_on{}_activation_{}' \
            .format(args.mod, args.optimizer,
                    args.loss_policy,
                    args.learning_rate,
                    args.batch_size,
                    args.end_to_end,
                    args.use_cholesky,
                    args.nclasses,
                    args.pretrained,
                    args.clas,
                    args.mask_percentage,
                    args.flip_on,
                    args.activation_layer)
    

    train_loader, valid_loader, valid_idx = get_loader(args.num_train,
                                                       args.json_file, 'Labels/lanes_ordered.json',
                                                       args.image_dir, 
                                                       args.gt_dir,
                                                       args.flip_on, args.batch_size, args.val_batch_size,
                                                       shuffle=True, num_workers=args.nworkers,
                                                       end_to_end=args.end_to_end,
                                                       resize=args.resize,
                                                       nclasses=args.nclasses,
                                                       split_percentage=args.split_percentage)

    test_loader = get_testloader(args.test_dir, args.val_batch_size, args.nworkers)

    # Define network
    model = Net(args)
    define_init_weights(model, args.weight_init)

    if not args.no_cuda:
        # Load model on gpu before passing params to optimizer
        model = model.cuda()

    # Define optimizer and scheduler
    optimizer = define_optim(args.optimizer, model.parameters(),
                             args.learning_rate, args.weight_decay)
    scheduler = define_scheduler(optimizer, args)


    # Define loss criteria for multiple tasks
    criterion, criterion_seg = define_loss_crit(args)
    criterion_horizon = nn.BCEWithLogitsLoss().cuda()
    criterion_line_class = nn.BCEWithLogitsLoss().cuda()

    # Name
    global crit_string
    if args.loss_policy == 'area' and args.end_to_end:
        crit_string = 'AREA**2' 
    elif args.loss_policy == 'backproject' and args.end_to_end:
        crit_string = 'MSE' 
    else:
        crit_string = 'ENTROPY' 
    if args.clas:
        crit_string = 'TOT LOSS' 

    # Logging setup
    best_epoch = 0
    lowest_loss = np.inf
    losses_valid = np.inf
    highest_score = 0
    log_file_name = 'log_train_start_0.txt'
    args.save_path = os.path.join(args.save_path, save_id)
    mkdir_if_missing(args.save_path)
    mkdir_if_missing(os.path.join(args.save_path, 'example/'))
    mkdir_if_missing(os.path.join(args.save_path, 'example/train'))
    mkdir_if_missing(os.path.join(args.save_path, 'example/valid'))
    mkdir_if_missing(os.path.join(args.save_path, 'example/pretrain'))
    mkdir_if_missing(os.path.join(args.save_path, 'example/testset'))

    # Computes the file with lane data of the validation set
    validation_set_path = os.path.join(args.save_path , 'validation_set.json')
    load_valid_set_file_all(valid_idx, validation_set_path, args.image_dir) 
    global valid_set_labels
    global val_set_path
    global ls_result_path
    valid_set_labels = [json.loads(line) for line in open(validation_set_path).readlines()]
    val_set_path = os.path.join(args.save_path, 'validation_set_dst.json')
    ls_result_path = os.path.join(args.save_path, 'ls_result.json')

    # Train, evaluate or resume
    args.resume = first_run(args.save_path)
    if args.resume and not args.test_mode and not args.evaluate:
        path = os.path.join(args.save_path, 'checkpoint_model_epoch_{}.pth.tar'.format(
            int(args.resume)))
        if os.path.isfile(path):
            log_file_name = 'log_train_start_{}.txt'.format(args.resume)
            # Redirect stdout
            sys.stdout = Logger(os.path.join(args.save_path, log_file_name))
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(path)
            args.start_epoch = checkpoint['epoch']
            lowest_loss = checkpoint['loss']
            best_epoch = checkpoint['best epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            log_file_name = 'log_train_start_0.txt'
            # Redirect stdout
            sys.stdout = Logger(os.path.join(args.save_path, log_file_name))
            print("=> no checkpoint found at '{}'".format(path))

    # Only evaluate
    elif args.evaluate:
        skip = get_flags()
        files = glob.glob(os.path.join(args.save_path, 'model_best*'))
        if len(files) == 0:
            print('No checkpoint found!')
        else:
            best_file_name = files[0]
            if os.path.isfile(best_file_name):
                sys.stdout = Logger(os.path.join(args.save_path, 'Evaluate.txt'))
                print("=> loading checkpoint '{}'".format(best_file_name))
                checkpoint = torch.load(best_file_name)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                print("=> no checkpoint found at '{}'".format(best_file_name))

        # validate(valid_loader, model, criterion, criterion_seg, 
                # criterion_line_class, criterion_horizon)

        if args.clas:
            test_model(test_loader, model, 
                       criterion, 
                       criterion_seg, 
                       criterion_line_class, 
                       criterion_horizon, args)
        return

    # Start training from clean slate
    else:
        # Redirect stdout
        sys.stdout = Logger(os.path.join(args.save_path, log_file_name))

    # INIT MODEL
    print(40*"="+"\nArgs:{}\n".format(args)+40*"=")
    print("Init model: '{}'".format(args.mod))
    print("Number of parameters in model {} is {:.3f}M".format(
        args.mod.upper(), sum(tensor.numel() for tensor in model.parameters())/1e6))

    # Define activation for classification branch
    if args.clas:
        Sigm = nn.Sigmoid()

    # Start training and validation for nepochs
    for epoch in range(args.start_epoch, args.nepochs):
        print("\n => Start train set for EPOCH {}".format(epoch + 1))
        print("Saving to: ", args.save_path)
        # Adjust learning rate
        if args.lr_policy is not None and args.lr_policy != 'plateau':
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            print('lr is set to {}'.format(lr))

        skip = get_flags(epoch)

        # Define container objects to keep track of multiple losses/metrics
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        rmse_metric = AverageMeter()
        losses_skip = AverageMeter()

        # Specfiy operation modus
        model.train()

        # compute timing
        end = time.time()

        # Start training loop
        for i, (input, gt, lanes, idx, gt_line, gt_horizon, valid_points) in tqdm(enumerate(train_loader)):
            # Time dataloader
            data_time.update(time.time() - end)

            # Reset coordinates
            x_cal0, x_cal1, x_cal2, x_cal3 = [None]*4

            # Put inputs on gpu if possible
            if not args.no_cuda:
                input, lanes = input.cuda(), lanes.cuda()
                valid_points = valid_points.cuda()
                gt = gt.cuda().squeeze(1)
            assert lanes.size(1) == 4
            gt0, gt1, gt2, gt3 = lanes[:, 0, :], lanes[:, 1, :], lanes[:, 2, :], lanes[:, 3, :]

            # Skip LSQ layer to make sure matrix cannot be singular
            # TODO check if this is really necessary
            if skip:
                output_net = model(input, gt_line, args.end_to_end, early_return=True)
                loss = criterion_seg(output_net, gt)
                # Setup backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses_skip.update(loss.item(), input.size(0))
                # Plot
                if (i + 1) % args.save_freq == 0:
                    img = input[0].permute(1, 2, 0).data.cpu().numpy()
                    gt_orig = gt[0].data.cpu().numpy()
                    _, out = torch.max(output_net[0], dim=0)
                    out = out.data.cpu().numpy()
                    img = np.clip(img, 0, 1)
                    fig = plt.figure()
                    ax1 = fig.add_subplot(311)
                    ax2 = fig.add_subplot(312)
                    ax3 = fig.add_subplot(313)
                    ax1.imshow(img)
                    ax2.imshow(gt_orig)
                    ax3.imshow(out)
                    fig.savefig(args.save_path + '/example/pretrain/idx-{}_batch-{}'.format(0, i))
                    plt.clf()
                    plt.close(fig)

                # Skip rest
                continue

            # Run model
            try:
                beta0, beta1, beta2, beta3, weightmap_zeros, \
                output_net, outputs_line, outputs_horizon, output_seg = model(input, gt_line, args.end_to_end, gt=gt)
            except RuntimeError as e:
                print("Batch with idx {} skipped due to singular matrix".format(idx.numpy()))
                print(e)
                continue

            # Compute losses on parameters or on segmentation
            if args.end_to_end:
                loss_left, x_cal0 = criterion(beta0, gt0, valid_points[:, 0])
                loss_right, x_cal1 = criterion(beta1, gt1, valid_points[:, 1])
                if args.nclasses > 3:
                    # add losses of further lane lines
                    loss_left1, x_cal2 = criterion(beta2, gt2, valid_points[:, 2])
                    loss_right1, x_cal3 = criterion(beta3, gt3, valid_points[:, 3])
                    loss_left += loss_left1
                    loss_right += loss_right1
                # average loss over lanes
                loss = (loss_left + loss_right) / args.nclasses
            else:
                loss = criterion_seg(output_net, gt)
                with torch.no_grad():
                    loss_left, x_cal0 = criterion(beta0, gt0, valid_points[:, 0])
                    loss_right, x_cal1 = criterion(beta1, gt1, valid_points[:, 1])
                    if args.nclasses > 3:
                        # add losses of further lane lines
                        loss_left1, x_cal2 = criterion(beta2, gt2, valid_points[:, 2])
                        loss_right1, x_cal3 = criterion(beta3, gt3, valid_points[:, 3])
                        loss_left += loss_left1
                        loss_right += loss_right1
                    loss_metric = (loss_left + loss_right) / args.nclasses
                    rmse_metric.update(loss_metric.item(), input.size(0))

            # Horizon task & Line classification task
            if args.clas:
                gt_horizon, gt_line = gt_horizon.cuda(), \
                                      gt_line.cuda()
                loss_horizon = criterion_horizon(outputs_horizon, gt_horizon).double()
                loss_line = criterion_line_class(outputs_line, gt_line).double()
                loss = loss*args.weight_fit + (loss_line + loss_horizon)*args.weight_class
            else:
                line_pred = gt_line

            # Update loss
            losses.update(loss.item(), input.size(0))

            # Clip gradients (usefull for instabilities or mistakes in ground truth)
            if args.clip_grad_norm != 0:
                nn.utils.clip_grad_norm(model.parameters(), args.clip_grad_norm)

            # Setup backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Time trainig iteration
            batch_time.update(time.time() - end)
            end = time.time()

            # Print info
            if (i + 1) % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.8f} ({loss.avg:.8f})\t'
                      'rmse_metric {rmse.val:.8f} ({rmse.avg:.8f})'.format(
                       epoch+1, i+1, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, rmse=rmse_metric))

            # Plot weightmap and curves
            if (i + 1) % args.save_freq == 0:
                save_weightmap('train', weightmap_zeros, x_cal0, x_cal1, x_cal2, x_cal3,
                               gt0, gt1, gt2, gt3, gt, 0, i, input,
                               args.no_ortho, args.resize, args.save_path, args.nclasses, args.no_mapping)

        print("===> Average {}-loss on training set is {:.8f}".format(crit_string, losses.avg))
        if not skip:
            losses_valid, acc_hor_tot, acc_line_tot, rmse_metric_valid = validate(valid_loader,
                                                                                  model, criterion,
                                                                                  criterion_seg, 
                                                                                  criterion_line_class,
                                                                                  criterion_horizon,
                                                                                  epoch)
            print("===> Average {}-loss on validation set is {:.8f}".format(crit_string, losses_valid))
        else:
            print("===> Average segmentation-loss on training set is {:.8f}".format(losses_skip.avg))
        if not args.end_to_end and not skip:
            print("===> Average rmse on training set is {:.8f}".format(rmse_metric.avg))
            print("===> Average rmse on validation set is {:.8f}".format(rmse_metric_valid))

        if args.clas and len(valid_loader) != 0 :
            print("===> Average HORIZON ACC on val is {:.8}".format(acc_hor_tot))
            print("===> Average LINE ACC on val is {:.8}".format(acc_line_tot))

        print("===> Last best {}-loss was {:.8f} in epoch {}".format(
            crit_string, lowest_loss, best_epoch))

        total_score = losses_valid

        # TODO get acc
        if args.clas:
            metric = test_model(test_loader, model, 
                       criterion, 
                       criterion_seg, 
                       criterion_line_class, 
                       criterion_horizon, args)
            total_score = metric


        # Adjust learning_rate if loss plateaued
        if args.lr_policy == 'plateau':
            scheduler.step(total_score)
            lr = optimizer.param_groups[0]['lr']
            print('LR plateaued, hence is set to {}'.format(lr))

        # File to keep latest epoch
        with open(os.path.join(args.save_path, 'first_run.txt'), 'w') as f:
            f.write(str(epoch))
        # Save model
        to_save = False
        if total_score > highest_score:
            to_save = True
            best_epoch = epoch+1
            highest_score = total_score
        save_checkpoint({
            'epoch': epoch + 1,
            'best epoch': best_epoch,
            'arch': args.mod,
            'state_dict': model.state_dict(),
            'loss': lowest_loss,
            'optimizer': optimizer.state_dict()}, to_save, epoch)


def validate(loader, model, criterion, criterion_seg, 
            criterion_line_class, criterion_horizon, epoch=0):

    # Define container to keep track of metric and loss
    losses = AverageMeter()
    acc_hor_tot = AverageMeter()
    acc_line_tot = AverageMeter()
    rmse_metric_valid = AverageMeter()

    # Evaluate model
    model.eval()

    # Only forward pass, hence no gradients/updates needed
    with torch.no_grad():
        
        # Start validation loop
        for i, (input, gt, lanes, idx, gt_line, gt_horizon, index, valid_points) in tqdm(enumerate(loader)):

            # Reset coordinates
            x_cal0, x_cal1, x_cal2, x_cal3 = [None]*4

            # Put inputs on gpu if possible
            if not args.no_cuda:
                input, lanes = input.cuda(), lanes.cuda()
                input = input.float()
                valid_points = valid_points.cuda()
                gt = gt.cuda().squeeze(1)
            assert lanes.size(1) == 4
            gt0, gt1, gt2, gt3 = lanes[:, 0, :], lanes[:, 1, :], lanes[:, 2, :], lanes[:, 3, :]

            # Run model
            try:
                beta0, beta1, beta2, beta3, weightmap_zeros, \
                output_net, outputs_line, outputs_horizon, output_seg = model(input, gt_line, args.end_to_end, gt = gt)
            except RuntimeError as e:
                print("Batch with idx {} skipped due to singular matrix".format(idx.numpy()))
                print(e)
                continue

            # Compute losses on parameters or on segmentation
            if args.end_to_end:
                loss_left, x_cal0 = criterion(beta0, gt0, valid_points[:, 0])
                loss_right, x_cal1 = criterion(beta1, gt1, valid_points[:, 1])
                if args.nclasses > 3:
                    # add losses of further lane lines
                    loss_left1, x_cal2 = criterion(beta2, gt2, valid_points[:, 2])
                    loss_right1, x_cal3 = criterion(beta3, gt3, valid_points[:, 3])
                    loss_left += loss_left1
                    loss_right += loss_right1
                # average loss over lanes
                loss = (loss_left + loss_right) / args.nclasses
            else:
                loss = criterion_seg(output_net, gt)
                with torch.no_grad():
                    loss_left, x_cal0 = criterion(beta0, gt0, valid_points[:, 0])
                    loss_right, x_cal1 = criterion(beta1, gt1, valid_points[:, 1])
                    if args.nclasses > 3:
                        # add losses of further lane lines
                        loss_left1, x_cal2 = criterion(beta2, gt2, valid_points[:, 2])
                        loss_right1, x_cal3 = criterion(beta3, gt3, valid_points[:, 3])
                        loss_left += loss_left1
                        loss_right += loss_right1
                    loss_metric = (loss_left + loss_right) / args.nclasses
                    rmse_metric_valid.update(loss_metric.item(), input.size(0))

            # Update losses
            losses.update(loss.item(), input.size(0))

            # Horizon task & Line classification task
            if args.clas:
                gt_horizon, gt_line = gt_horizon.cuda(), \
                                      gt_line.cuda()
                horizon_pred = torch.round(nn.Sigmoid()(outputs_horizon))
                acc = torch.eq(horizon_pred, gt_horizon)
                acc_hor = torch.sum(acc).float()/(args.resize*args.val_batch_size)
                acc_hor_tot.update(acc_hor.item())
                line_pred = torch.round(nn.Sigmoid()(outputs_line))
                acc = torch.eq(line_pred, gt_line)
                acc_line = torch.sum(acc).float()/(args.nclasses*args.val_batch_size)
                acc_line_tot.update(acc_line.item())
                loss_horizon = criterion_horizon(outputs_horizon, gt_horizon)
                loss_line = criterion_line_class(outputs_line, gt_line)
                loss = loss*args.weight_fit + (loss_line + loss_horizon)*args.weight_class
            else:
                line_pred = gt_line

            # Print info
            if (i + 1) % args.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Loss {loss.val:.8f} ({loss.avg:.8f})\t'
                          'rmse {rmse_metric.val:.8f} ({rmse_metric.avg:.8f})'.format(
                           i+1, len(loader), loss=losses, rmse_metric=rmse_metric_valid))

            # Plot weightmap and curves
            if (i + 1) % 25 == 0:
                save_weightmap('valid', weightmap_zeros, x_cal0, x_cal1, x_cal2, x_cal3,
                               gt0, gt1, gt2, gt3, gt, 0, i, input,
                               args.no_ortho, args.resize, args.save_path, args.nclasses, args.no_mapping)

        # Print statistic about epoch
        if args.evaluate:
            print("===> Average {}-loss on validation set is {:.8}".format(crit_string, 
                                                                           losses.avg))
            if args.clas and len(loader) != 0:
                print("===> Average HORIZON ACC on val is {:.8}".format(acc_hor_tot.avg))
                print("===> Average LINE ACC on val is {:.8}".format(acc_hor_tot.avg))

        return losses.avg, acc_hor_tot.avg, acc_line_tot.avg, rmse_metric_valid.avg


def save_checkpoint(state, to_copy, epoch):
    filepath = os.path.join(args.save_path, 'checkpoint_model_epoch_{}.pth.tar'.format(epoch))
    torch.save(state, filepath)
    if to_copy:
        if epoch > 0:
            lst = glob.glob(os.path.join(args.save_path, 'model_best*'))
            if len(lst) != 0:
                os.remove(lst[0])
        shutil.copyfile(filepath, os.path.join(args.save_path, 
            'model_best_epoch_{}.pth.tar'.format(epoch)))
        print("Best model copied")
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(args.save_path, 
                'checkpoint_model_epoch_{}.pth.tar'.format(epoch-1))
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)


if __name__ == '__main__':
    main()
