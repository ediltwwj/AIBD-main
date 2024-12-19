import copy

import torch
import config
import torchattacks
import torch.nn as nn
import torch.nn.functional as F
import models.resnet
import models.wresnet
import torch.backends.cudnn as cudnn
from utils import *
from torch.utils.data import Dataset, DataLoader


def make_criterion(alpha=0.5, T=4.0, kd_mode='cse'):
    def criterion(outputs, targets, labels):
        if kd_mode == 'cse':
            _p = F.log_softmax(outputs / T, dim=1)
            _q = F.softmax(targets / T, dim=1)
            _soft_loss = -torch.mean(torch.sum(_q * _p, dim=1))
        elif kd_mode == 'mse':
            _p = F.softmax(outputs / T, dim=1)
            _q = F.softmax(targets / T, dim=1)
            _soft_loss = nn.MSELoss()(_p, _q) / 2
        elif kd_mode == 'kl':
            _p = F.log_softmax(outputs / T, dim=1)
            _q = F.log_softmax(targets / T, dim=1)
            _soft_loss = nn.KLDivLoss(reduction='mean')(_p, _q)
        else:
            raise NotImplementedError()

        _soft_loss = _soft_loss * T * T
        _hard_loss = F.cross_entropy(outputs, labels)
        loss = alpha * _soft_loss + (1. - alpha) * _hard_loss
        return loss

    return criterion


def get_num_class(args):
    if args.dataset == 'CIFAR10' or args.dataset == 'cifar10':
        args.num_class = 10
    elif args.dataset == 'CIFAR100' or args.dataset == 'cifar100':
        args.num_class = 100
    elif args.dataset == 'GTSRB' or args.dataset == 'gtsrb':
        args.num_class = 43
    elif args.dataset == 'ImageNet-Subset-20' or args.dataset == 'imagenet-subset-20':
        args.num_class = 20
    else:
        raise NotImplementedError()


def get_base_network(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    get_num_class(args)

    if args.base_network == 'resnet18':
        print("==> Initing Base Model Resnet18...")
        net = models.resnet.resnet18(pretrained=args.use_pretrain_model)
        net.fc = nn.Linear(512, args.num_class)
    elif args.base_network == 'resnet34':
        print("==> Initing Base Model Resnet34...")
        net = models.resnet.resnet34(pretrained=args.use_pretrain_model)
        net.fc = nn.Linear(512, args.num_class)
    elif args.base_network == 'wrn-16-2':
        print("==> Initing Base Model Wrn-16-2...")
        net = models.wresnet.WideResNet(depth=16, num_classes=args.num_class, widen_factor=2)
    elif args.base_network == 'wrn-16-1':
        print("==> Initing Base Model Wrn-16-1...")
        net = models.wresnet.WideResNet(depth=16, num_classes=args.num_class, widen_factor=1)
    else:
        raise NotImplementedError()

    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    return net


def base_train_epoch(base_net, train_loader, device, criterion, optimizer, scheduler, args):
    base_net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets, index) in enumerate(train_loader):
        inputs, targets, index = inputs.to(device), targets.to(device), index.to(device)

        optimizer.zero_grad()
        outputs = base_net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx,
                     len(train_loader), 'TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d)' %
                     (train_loss / (batch_idx + 1), 100. * correct / total,
                      correct, total))
    scheduler.step()

    return 100. * correct / total


def val_epoch(net, clean_test_loader, bd_test_loader, device, criterion, args):
    net.eval()

    # Clean Val
    clean_val_loss = 0
    clean_correct = 0
    clean_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(clean_test_loader):
            inputs, targets, index = inputs.to(device), targets.to(device), index.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            clean_val_loss += loss.item()
            _, predicted = outputs.max(1)
            clean_total += targets.size(0)
            clean_correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx,
                         len(clean_test_loader), 'Clean ValLoss: %.3f | ValAcc: %.3f%% (%d/%d)' %
                         (clean_val_loss / (batch_idx + 1), 100. * clean_correct / clean_total,
                          clean_correct, clean_total))

    clean_acc = 100. * clean_correct / clean_total

    # Backdoor Val
    bd_val_loss = 0
    bd_correct = 0
    bd_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(bd_test_loader):
            inputs, targets, index = inputs.to(device), targets.to(device), index.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            bd_val_loss += loss.item()
            _, predicted = outputs.max(1)
            bd_total += targets.size(0)
            bd_correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx,
                         len(bd_test_loader), 'Bd ValLoss: %.3f | ValAsr: %.3f%% (%d/%d)' %
                         (bd_val_loss / (batch_idx + 1), 100. * bd_correct / bd_total,
                          bd_correct, bd_total))
    bd_asr = 100. * bd_correct / bd_total

    return clean_acc, bd_asr
