import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import *
from getInit import *


def pgd_attack_new(net, loss_fn, image, label, args):
    epsilon, alpha, num_iterations = args.adv_epsilon, args.adv_alpha, args.adv_num_iterations
    perturbed_image = image.clone().detach().requires_grad_(True).to('cuda')

    output = net(perturbed_image)
    _, predicted = output.max(1)
    predict_label = predicted.item()

    count = 0
    for _ in range(num_iterations):
        output = net(perturbed_image)
        _, predicted = output.max(1)
        loss = loss_fn(output, label)

        net.zero_grad()

        if predicted.item() == predict_label:
            count = count + 1
        else:
            break
        loss.backward()

        with torch.no_grad():
            perturbed_image_grad = alpha * perturbed_image.grad.sign()
            perturbed_image += perturbed_image_grad
            perturbed_image = torch.max(torch.min(perturbed_image, image + epsilon), image - epsilon)
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
        perturbed_image.requires_grad_(True)

    return perturbed_image.detach(), count, predicted.item()


def count_adversarial_iter_nums(train_data, net, args, device='cuda'):
    """
        Carry out adversarial attacks to obtain adversarial samples, the number of adversarial attacks, and the adversarial labels
    """
    print("Count Adversarial Iter Nums Begin...")
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(dataset=train_data,
                              batch_size=1,
                              shuffle=False,
                              )

    perturbation_matrices_list = torch.empty(2, 3).to(device)
    count_list = []
    adv_label_list = []
    for batch_idx, (input, target, index) in enumerate(train_loader):
        input, target, index = input.to(device), target.to(device), index.to(device)
        adv_input, count, adv_label = pgd_attack_new(net, loss_fn, input, target, args)

        count_list.append(count)
        adv_label_list.append(adv_label)

        if (batch_idx + 1) % 5000 == 0:
            print("================={}================".format(batch_idx + 1))

    print(perturbation_matrices_list.size())
    print("Count Adversarial Iter Nums Finish...")
    return perturbation_matrices_list, count_list, adv_label_list


def guess_bdattack_label(count_list, bd_train_data, args, device='cuda'):
    """
        Identify potential backdoor label by the number of adversarial attacks
    """

    probability_idx = np.argsort(np.array(count_list))[::-1]
    label_idx = probability_idx[0:int(len(probability_idx) * args.label_rate)]

    bd_train_loader = DataLoader(dataset=bd_train_data,
                                 batch_size=1,
                                 shuffle=False,
                                 )

    stat_list = [0 for _ in range(args.num_class)]
    for batch_idx, (input, target, index) in enumerate(bd_train_loader):
        if index.item() in label_idx:
            stat_list[target.item()] += 1

    max_value = max(stat_list)
    guess_bd_label = stat_list.index(max_value)

    return guess_bd_label


def isloation_bd_data(bd_train_data, count_list, bd_perm, adv_label_list, guess_bd_label, ori_dict,
                      perturbation_matrices, args):
    """
        A purified dataset is obtained by presetting the poisoning rate and modifying the labels of potentially poisoned samples to adversarial labels
    """

    bd_train_loader = DataLoader(dataset=bd_train_data,
                                 batch_size=1,
                                 shuffle=False,
                                 )

    bd_label_samples_num = 0
    bd_label_indexs_list = []
    bd_label_iter_nums_list = []
    for batch_idx, (input, target, index) in enumerate(bd_train_loader):
        if target.item() == guess_bd_label:
            bd_label_samples_num += 1
            bd_label_indexs_list.append(index.item())
            bd_label_iter_nums_list.append(count_list[index.item()])

    data_path_new_list = []

    for ratio in args.iso_ratio:
        threshold_idx = int(args.num_data * ratio)
        if threshold_idx <= bd_label_samples_num:
            temp_list = sorted(bd_label_iter_nums_list, reverse=True)
            threshold_nums = temp_list[threshold_idx - 1]

            new_example_list = []
            cnt = 0
            bd_cnt = 0
            for batch_idx, (input, target, index) in enumerate(bd_train_loader):
                if target.item() == guess_bd_label and count_list[index.item()] >= threshold_nums:
                    # <原样本, 原标签> ==> <原样本, 对抗标签>
                    input = input.squeeze()
                    input = np.transpose((input * 255).cpu().numpy(), (1, 2, 0)).astype('uint8')
                    target = adv_label_list[index.item()]

                    cnt += 1
                    if index.item() in bd_perm:
                        bd_cnt += 1
                else:
                    input = input.squeeze()
                    input = np.transpose((input * 255).cpu().numpy(), (1, 2, 0)).astype('uint8')
                    target = target.squeeze().cpu().numpy()

                new_example_list.append((input, target))

            data_path_new = ''
            if args.iso_save == 1:
                print("==>Saving new data...")
                data_path_new = os.path.join(args.new_data_path,
                                             "{}_{}_{}_{}_data_new_examples.npy".format(args.dataset,
                                                                                        args.base_network,
                                                                                        args.trigger_type,
                                                                                        ratio * 100.0))
                if not os.path.isdir('new_data'):
                    os.mkdir('new_data')
                np.save(data_path_new, new_example_list)

                data_path_new_list.append(data_path_new)

                print('==>Finish save new data ratio {}%...'.format(ratio * 100))
            else:
                print("==>Not save new data...")

    return data_path_new_list


class CustomException(Exception):
    """
        Custom exception classes
    """

    def __init__(self, message):
        self.message = message


def learning_rate_new_train(optimizer, epoch, args):
    if epoch >= 0 and epoch < 5:
        lr = 0.1
    elif epoch >= 5 and epoch < 10:
        lr = 0.01
    elif epoch >= 10 and epoch < 15:
        lr = 0.001
    else:
        lr = 0.0001
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def __new_train_epoch(net, isolation_data_loader, device, optimizer, criterion, args):
    net.train()
    new_train_loss = 0

    for batch_idx, (inputs, targets, index) in enumerate(isolation_data_loader):
        inputs, targets, index = inputs.to(device), targets.to(device), index.to(device)
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        new_train_loss += loss.item()

        progress_bar(batch_idx, len(isolation_data_loader),
                     'New Train Loss : %.3f' % (new_train_loss / (batch_idx + 1)))


def _base_train(train_loader, clean_test_loader, bd_test_loader, args):
    print('==> Base Model Train Begin...')
    start_epoch = 0
    best_acc, best_asr = 0.0, 0.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ori_base_net = get_base_network(args)
    ori_best_net = copy.deepcopy(ori_base_net)

    base_criterion = nn.CrossEntropyLoss()
    base_optimizer = optim.SGD(ori_base_net.parameters(), lr=args.base_train_lr, momentum=0.9, weight_decay=5e-4)
    base_scheduler = optim.lr_scheduler.MultiStepLR(base_optimizer, args.base_train_scheduler_milestones,
                                                    args.base_train_scheduler_lambda
                                                    )

    for epoch_idx in range(start_epoch, args.base_train_epoch):
        print('\nEpoch: %d' % epoch_idx)
        train_acc = base_train_epoch(ori_base_net, train_loader, device, base_criterion, base_optimizer, base_scheduler,
                                     args)
        acc, asr = val_epoch(ori_base_net, clean_test_loader, bd_test_loader, device, base_criterion, args)

        if acc > best_acc:
            print("Saving best base model...")
            best_acc, best_asr = acc, asr
            ori_best_net = copy.deepcopy(ori_base_net)

    return ori_best_net, best_acc, best_asr


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    bd_perm, ori_dict, bd_train_data, bd_train_loader = get_backdoor_loader(args)
    clean_test_loader, bd_test_loader = get_test_loader(args)

    print('\n---------------Normal Train Step---------------')
    ori_best_net, ori_acc, ori_asr = _base_train(bd_train_loader, clean_test_loader, bd_test_loader, args)

    print('\n---------------Count Adversarial Iter_Nums---------------')
    perturbation_matrices, count_list, adv_label_list = count_adversarial_iter_nums(bd_train_data,
                                                                                    ori_best_net, args)

    print('\n---------------Get New Data---------------')
    guess_bd_label = guess_bdattack_label(count_list, bd_train_data, args)
    data_path_new_list = isloation_bd_data(
        bd_train_data, count_list, bd_perm, adv_label_list, guess_bd_label, ori_dict,
        perturbation_matrices, args)

    if args.dataset == "CIFAR10":
        tf_new_train = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        tf_new_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    for i in range(len(data_path_new_list)):
        new_data = np.load(data_path_new_list[i], allow_pickle=True)
        new_data_tf = Dataset_npy(full_dataset=new_data, transform=tf_new_train)
        new_data_loader = DataLoader(dataset=new_data_tf, batch_size=args.train_bs, shuffle=True)

        aibd_base_net = copy.deepcopy(ori_best_net)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(aibd_base_net.parameters(), lr=args.new_train_lr, momentum=0.9, weight_decay=5e-4)

        print('\n---------------New Train Step---------------')
        aibd_acc, aibd_asr = ori_acc, ori_asr
        temp_acc, temp_asr = 0.0, 0.0
        for epoch_idx in range(0, args.new_train_epoch):
            print("\nnew train epoch:{}".format(epoch_idx))
            learning_rate_new_train(optimizer, epoch_idx, args)
            __new_train_epoch(aibd_base_net, new_data_loader, device, optimizer, criterion, args)
            temp_acc, temp_asr = val_epoch(aibd_base_net, clean_test_loader, bd_test_loader, device, criterion, args)

            # When the accuracy of the test is too different, it indicates that catastrophic forgetting occurs, and the top-q strategy is adopted
            if epoch_idx == 4 and temp_acc < ori_acc - args.top_q_threadhold * 2.0:   # args.top_q_threadhold = 4.0
            # if epoch_idx == 5 and temp_acc < ori_acc - args.top_q_threadhold * 2.0:   # args.top_q_threadhold = 3.5
                print("Catastrophic Forgetting Occurs...")
                break
            else:
                aibd_acc, aibd_asr = temp_acc, temp_asr

        if temp_acc >= ori_acc - args.top_q_threadhold:
            print("Saving Best AIBD model {}%...".format(args.iso_ratio[i] * 100.0))
            aibd_acc, aibd_asr = temp_acc, temp_asr
            break

    return ori_acc, ori_asr, aibd_acc, aibd_asr, guess_bd_label


if __name__ == "__main__":
    args = config.get_arguments().parse_args()

    content = ""
    for tp in args.trigger_list:
        args.trigger_type = tp
        args.bd_ori_dict = True

        print("Attack method:{}".format(args.trigger_type))

        ori_acc, ori_asr, aibd_acc, aibd_asr, guess_bd_label = train(args)
        content += "{}\nori_acc: {}, ori_asr: {}\n".format(args.trigger_type, ori_acc, ori_asr)
        content += "aibd_acc: {}, aibd_asr: {}\n".format(aibd_acc, aibd_asr)
        content += "guess_bd_label: {}\n".format(guess_bd_label)

        print(content)

with open('metric/{}_{}_{}.txt'.format(args.base_network, args.dataset, args.poison_rate),
          'w+') as f:
    f.write(content)
