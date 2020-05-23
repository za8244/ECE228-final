# Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py

import os
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from inception import *

import loader_mammal

class Params:
    num_classes = 235 #change here
    epochs = 50 # 70 epochs
    start_epoch = 0
    batch_size = 16
    lr = 0.0045
    lr_decay = 0.94
    epoch_decay = 4
    momentum = 0.9
    weight_decay = 1e-4
    print_freq = 100
    resume = ''  # set this to path of model to resume training
    #resume = '/home/macaodha/Projects/inat2018/code/model_best.pth.tar'
    #train_file = '/media/derek/65fcdb8d-2222-4ffc-93e2-9dcb525bcf8d/data/train2018.json' #set data path here
    #val_file = '/media/derek/65fcdb8d-2222-4ffc-93e2-9dcb525bcf8d/data/val2018.json'
    data_root = '/media/derek/65fcdb8d-2222-4ffc-93e2-9dcb525bcf8d/images_mammal/Mammalia'
    # set evaluate to True to run the test set
    evaluate = False
    best_prec3 = 0.0  # store current best top 3


def main():
    global args, best_prec3
    best_prec3 = 0.0
    args = Params()
    #===============data====================================
    train_dataset = loader_mammal.INAT(args.data_root, is_train=True)
    val_dataset = loader_mammal.INAT(args.data_root, is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True) #pin_memory?
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  pin_memory=True)
    #===============model====================================
    print("Using pre-trained inception_v3")
    model = inception_v3(pretrained=True)
    model.fc = nn.Linear(2048, args.num_classes)
    model.aux_logits = False # what is this?
    model = model.cuda()
    #===============training parameters======================
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec3 = checkpoint['best_prec3']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True # for efficiency

    if args.evaluate:
        ###To do###
        print("testing")
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec3 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec3 > best_prec3
        best_prec3 = max(prec3, best_prec3)
        
        save_checkpoint({
            'epoch': epoch + 1,
            #'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec3': best_prec3,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    print('Epoch:{0}'.format(epoch))
    print('Itr\t\tTime\t\tLoss\t\tPrec@1\t\tPrec@3')
    for i, rawData in enumerate(train_loader):
        input = rawData[0].cuda()
        #target = target.cuda(async=True)
        target = rawData[1].cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        #print("output shape: ",output.shape) # (batchsize, nClasses)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        #losses.update(loss.data[0], input.size(0)) #modify here
        losses.update(loss.data.cpu(), input.size(0)) #modify here
        top1.update(prec1[0], input.size(0))
        top3.update(prec3[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('[{0}/{1}]\t'
                '{batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                '{loss.val:.3f} ({loss.avg:.3f})\t'
                '{top1.val:.2f} ({top1.avg:.2f})\t'
                '{top3.val:.2f} ({top3.avg:.2f})'.format(
                i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1, top3=top3))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    im_ids = []

    print('Validate:\tTime\t\tLoss\t\tPrec@1\t\tPrec@3')
    for i, rawData in enumerate(val_loader):

        input = rawData[0].cuda()
        target = rawData[1].cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.data.cpu(), input.size(0)) #modify here
        top1.update(prec1[0], input.size(0))
        top3.update(prec3[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('[{0}/{1}]\t'
                  '{batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  '{loss.val:.3f} ({loss.avg:.3f})\t'
                  '{top1.val:.2f} ({top1.avg:.2f})\t'
                  '{top3.val:.2f} ({top3.avg:.2f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top3=top3))

    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
          .format(top1=top1, top3=top3))
    return top3.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print("\tSaving new best model")
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
