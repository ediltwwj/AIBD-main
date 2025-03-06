import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
from PIL import Image
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms, datasets
from tqdm import tqdm

import models.resnet
import models.wresnet
from networks.models import Generator  # dynamic attack Generator


class GTSRB(Dataset):
    def __init__(self, args, train, transforms=None):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(args.data_path, "GTSRB/Train")
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(args.data_path, "GTSRB/Test")
            self.images, self.labels = self._get_data_test_list()

        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        if self.transforms is not None:
            image = self.transforms(image)
        label = self.labels[index]
        return image, label, index


def get_train_loader(args):
    print('==> Preparing train data..')

    # transform list
    args.img_w, args.img_h = 32, 32
    rs = transforms.Resize((args.img_h, args.img_w))
    rc = transforms.RandomCrop(32, padding=4)
    rr = transforms.RandomRotation(10)
    rhf = transforms.RandomHorizontalFlip()
    rt = transforms.ToTensor()

    if (args.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root=args.data_path, train=True, download=True)
        tf_train = transforms.Compose([rt])

    elif (args.dataset == 'CIFAR100'):
        trainset = datasets.CIFAR100(root=args.data_path, train=True, download=True)
        tf_train = transforms.Compose([rt])
    elif (args.dataset == 'GTSRB'):
        gtsrb = GTSRB(args, train=True, transforms=transforms.Compose([transforms.Resize((args.img_h, args.img_w))]))
        trainset = []
        for i in tqdm(range(len(gtsrb))):
            img, label, index = gtsrb[i]
            trainset.append((img, label))
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        tf_train = transforms.Compose([rt])
    else:
        raise Exception('Invalid dataset')

    train_data = DatasetCL(args, full_dataset=trainset, transform=tf_train)
    train_loader = DataLoader(train_data, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers)

    return train_data, train_loader


def get_test_loader(args):
    print('==> Preparing test data..')
    if (args.dataset == 'CIFAR10'):
        testset = datasets.CIFAR10(root=args.data_path, train=False, download=True)
    elif (args.dataset == 'CIFAR100'):
        testset = datasets.CIFAR100(root=args.data_path, train=False, download=True)
    elif (args.dataset == 'GTSRB'):
        gtsrb = GTSRB(args, train=False, transforms=transforms.Compose([transforms.Resize((args.img_h, args.img_w))]))
        testset = []
        for i in tqdm(range(len(gtsrb))):
            img, label, index = gtsrb[i]
            testset.append((img, label))
    else:
        raise Exception('Invalid dataset')

    tf_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_data_clean = DatasetBD(args, full_dataset=testset, inject_portion=0, transform=tf_test, mode='test')
    test_data_bd = DatasetBD(args, full_dataset=testset, inject_portion=1, transform=tf_test, mode='test')

    # (apart from label 0) bad test data
    test_clean_loader = DataLoader(dataset=test_data_clean,
                                   batch_size=args.test_bs,
                                   shuffle=False,
                                   num_workers=args.num_workers,
                                   )
    # all clean test data
    test_bd_loader = DataLoader(dataset=test_data_bd,
                                batch_size=args.test_bs,
                                shuffle=False,
                                num_workers=args.num_workers,
                                )

    return test_clean_loader, test_bd_loader


def add_gaussian_noise(image, mean=0, std=10):
    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    return noisy_image


class GaussianNoise(object):
    def __init__(self, mean=0, std=10):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        noisy_image = add_gaussian_noise(image, self.mean, self.std)
        return noisy_image


def get_backdoor_loader(args):
    print('==> Preparing train data..')
    # transform list
    args.img_w, args.img_h = 32, 32
    rs = transforms.Resize((args.img_h, args.img_w))
    rc = transforms.RandomCrop(32, padding=4)
    rr = transforms.RandomRotation(10)
    rhf = transforms.RandomHorizontalFlip()
    rt = transforms.ToTensor()

    if (args.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root=args.data_path, train=True, download=True)
        tf_train = transforms.Compose([rt])
    elif (args.dataset == 'CIFAR100'):
        trainset = datasets.CIFAR100(root=args.data_path, train=True, download=True)
        tf_train = transforms.Compose([rt])
    elif (args.dataset == 'GTSRB'):
        gtsrb = GTSRB(args, train=True, transforms=transforms.Compose([transforms.Resize((args.img_h, args.img_w))]))
        trainset = []
        for i in tqdm(range(len(gtsrb))):
            img, label, index = gtsrb[i]
            trainset.append((img, label))
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        tf_train = transforms.Compose([rt])
    else:
        raise Exception('Invalid dataset')

    train_data_bd = DatasetBD(args, full_dataset=trainset, inject_portion=args.poison_rate, transform=tf_train,
                              mode='train')
    poison_perm = train_data_bd.getPerm()
    ori_dict = train_data_bd.get_Ori_Dict()
    train_bd_loader = DataLoader(dataset=train_data_bd,
                                 batch_size=args.train_bs,
                                 shuffle=True,  # True
                                 num_workers=args.num_workers,
                                 )
    if args.bd_ori_dict:
        return poison_perm, ori_dict, train_data_bd, train_bd_loader
    else:
        return poison_perm, train_data_bd, train_bd_loader


class Dataset_npy(torch.utils.data.Dataset):
    def __init__(self, full_dataset=None, transform=None):
        self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)
        # print(type(image), image.shape)
        return image, label, index

    def __len__(self):
        return self.dataLen


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class DatasetCL(Dataset):
    def __init__(self, args, full_dataset=None, transform=None):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=args.train_ratio)
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)

        return image, label, index

    def __len__(self):
        return self.dataLen

    def random_split(self, full_dataset, ratio):
        print('full_train:', len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
        print('train_size:', len(train_dataset), 'drop_size:', len(drop_dataset))

        return train_dataset


class DatasetBD(Dataset):
    def __init__(self, args, full_dataset, inject_portion, transform=None, mode="train", device=torch.device("cuda"),
                 distance=1):

        if args.trigger_type == 'wanet':
            if args.dataset == 'GTSRB':
                state_dict = torch.load(
                    'triggers/trigger_models/wanet/{}_all2one_{}_morph.pth.tar'.format(args.dataset.lower(),
                                                                                       args.base_network.lower()))
            else:
                state_dict = torch.load(
                    'triggers/trigger_models/wanet/{}_all2one_{}_morph.pth.tar'.format(args.dataset.lower(),
                                                                                       args.base_network.lower()))
            self.identity_grid = state_dict['identity_grid'].cpu()
            self.noise_grid = state_dict['noise_grid'].cpu()
        if args.trigger_type == 'dynamic':
            if args.dataset == 'GTSRB':
                state_dict_g = torch.load(
                    'triggers/trigger_models/dynamic/{}_all2one_{}_morph.pth.tar'.format(args.dataset.lower(),
                                                                                         args.base_network.lower()))
            else:
                state_dict_g = torch.load(
                    'triggers/trigger_models/dynamic/{}_all2one_{}_morph.pth.tar'.format(args.dataset.lower(),
                                                                                         args.base_network.lower()))
            self.netG = Generator(args)
            self.netG.load_state_dict(state_dict_g['netG'])
            self.netG.eval()
            self.netG.requires_grad_(False)

            if args.dataset == 'GTSRB':
                state_dict_m = torch.load(
                    'triggers/trigger_models/dynamic/mask/{}_all2one_{}_morph.pth.tar'.format(args.dataset.lower(),
                                                                                              args.base_network.lower()))
            else:
                state_dict_m = torch.load(
                    'triggers/trigger_models/dynamic/mask/{}_all2one_{}_morph.pth.tar'.format(args.dataset.lower(),
                                                                                              args.base_network.lower()))
            self.netM = Generator(args, out_channels=1)
            self.netM.load_state_dict(state_dict_m['netM'])
            self.netM.eval()
            self.netM.requires_grad_(False)

        self.mode = mode
        self.dataset, self.perm, self.ori_dict = self.addTrigger(full_dataset, inject_portion, mode, distance, args)
        self.device = device
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        img = self.transform(img)

        return img, label, item

    def __len__(self):
        return len(self.dataset)

    def getPerm(self):
        # return list
        return self.perm

    def get_Ori_Dict(self):
        # return dict
        return self.ori_dict

    def addNoise(self, dataset, args, noise_rate):
        num_sample_per_label = [0] * args.num_class
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            label = data[1]
            num_sample_per_label[label] += 1

        noise_num_sample_per_label = [int(i * noise_rate) for i in num_sample_per_label]

        dataset_ = list()
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            img = data[0]
            label = data[1]
            if noise_num_sample_per_label[label] > 0:
                img_tensor = transforms.ToTensor()(img)
                img_noise = add_gaussian_noise(img_tensor)
                img = transforms.ToPILImage()(img_noise)
                dataset_.append((img, label))
                noise_num_sample_per_label[label] -= 1
            else:
                dataset_.append((img, label))
        print("Finish add noise...")

        return dataset_

    def addTrigger(self, dataset, inject_portion, mode, distance, args):
        print("Generating " + mode + "bd Imgs")
        if args.trigger_type == 'cleanLabel' and mode == 'train':
            perm_init = []
            for i in tqdm(range(len(dataset))):
                data = dataset[i]
                if data[1] == args.target_label:
                    perm_init.append(i)
            perm = random.sample(perm_init, k=int(len(perm_init) * inject_portion * 5.0))
        else:
            perm_init = range(len(dataset))
            perm = random.sample(perm_init, k=int(len(perm_init) * inject_portion))

        # dataset
        dataset_ = list()

        cnt = 0
        ori_dict = {}
        for i in tqdm(range(len(dataset))):
            data = dataset[i]

            if args.target_type == 'all2one':

                if mode == 'train':
                    img = data[0]

                    if i in perm:
                        # select trigger
                        img = self.selectTrigger(img, args)

                        # change target
                        dataset_.append((img, args.target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))
                    ori_dict[i] = data[1]
                else:
                    if data[1] == args.target_label and inject_portion == 1.0:
                        continue

                    img = data[0]
                    if i in perm:
                        img = self.selectTrigger(img, args)

                        dataset_.append((img, args.target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # all2all attack
            elif args.target_type == 'all2all':

                if mode == 'train':
                    img = data[0]
                    if i in perm:

                        img = self.selectTrigger(img, args)
                        target_ = self._change_label_next(data[1], args)

                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))
                    ori_dict[i] = data[1]
                else:

                    img = data[0]
                    if i in perm:
                        img = self.selectTrigger(img, args)

                        target_ = self._change_label_next(data[1], args)
                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # clean label attack
            elif args.target_type == 'cleanLabel':

                if mode == 'train':
                    img = data[0]

                    if i in perm:
                        if data[1] == args.target_label:

                            img = self.selectTrigger(img, args)

                            dataset_.append((img, data[1]))
                            cnt += 1

                        else:
                            dataset_.append((img, data[1]))
                    else:
                        dataset_.append((img, data[1]))
                    ori_dict[i] = data[1]
                else:
                    if data[1] == args.target_label:
                        continue

                    img = data[0]
                    if i in perm:
                        img = self.selectTrigger(img, args)

                        dataset_.append((img, args.target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

        time.sleep(0.01)
        print("Injecting Over: " + str(cnt) + "Bad Imgs, " + str(len(dataset) - cnt) + "Clean Imgs")

        return dataset_, perm, ori_dict

    def _change_label_next(self, label, args):
        label_new = ((label + 1) % args.num_class)
        return label_new

    def selectTrigger(self, img, args):

        assert args.trigger_type in ['badnets', 'blend', 'sig', 'wanet', 'trojan', 'dynamic']

        if args.trigger_type == 'badnets':
            img = self._badnets(img, args.img_w, args.img_h, args.trig_w, args.trig_h)

        elif args.trigger_type == 'blend':
            img = self._blend(img, args.img_w, args.img_h)

        elif args.trigger_type == 'sig':
            img = self._sig(img, args, delta=20.0, frequency=6)

        elif args.trigger_type == 'wanet':
            img = self._wanet(img, args, s=0.5, k=4, grid_rescale=1)

        elif args.trigger_type == 'trojan':
            img = self._trojan(img, args)

        elif args.trigger_type == 'dynamic':  # input-aware attack
            img = self._dynamic(img, args)
        else:
            raise NotImplementedError

        return img

    def _badnets(self, img, width, height, trig_w, trig_h, distance=1):
        img_arr = np.array(img)

        # random
        # j = random.randint(0, width - trig_w - distance)
        # k = random.randint(0, height - trig_h - distance)
        # for x in range(j, j + trig_w):
        #     for y in range(k, k + trig_h):
        #         img_arr[x, y] = 255.0

        # fixed
        for j in range(width - distance - trig_w, width - distance):
            for k in range(height - distance - trig_h, height - distance):
                img_arr[j, k] = 255.0

        img = Image.fromarray(img_arr)
        return img

    def _blend(self, img, width, height, distance=1):
        alpha = 0.2
        img_arr = np.array(img)
        mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
        blend_img = (1 - alpha) * img_arr + alpha * mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        img = Image.fromarray(blend_img)
        return img

    def _sig(self, img, args, delta, frequency):
        img_arr = np.array(img)
        overlay = np.zeros(img_arr.shape, np.float64)
        _, m, _ = overlay.shape
        for i in range(m):
            overlay[:, i] = delta * np.sin(2 * np.pi * i * frequency / m)
        img_arr = np.clip(overlay + img_arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
        return img

    def _wanet(self, img, args, s, k, grid_rescale):
        img_arr = np.array(img)  # H,W,C
        img_torch = torch.from_numpy(img_arr).unsqueeze(0).to(torch.float32).permute(0, 3, 1, 2)
        grid_temps = (self.identity_grid + s * self.noise_grid / args.img_h) * grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)
        img_torch = F.grid_sample(img_torch, grid_temps.repeat(1, 1, 1, 1), align_corners=True).permute(0, 2, 3, 1)
        img_arr = np.array(img_torch.squeeze(0) * 255.0).astype(np.uint8)  # 前面clamp到(-1,1)
        img = Image.fromarray(img_arr)

        return img

    def _trojan(self, img, args):
        img_arr = np.array(img)
        if args.dataset == 'CIFAR10' or args.dataset == 'cifar10':
            trg = np.load('triggers/trojan_cifar10.npz')['x']
            trg = np.transpose(trg, (1, 2, 0))
        else:
            pass
        img_arr = np.clip((img_arr + trg).astype('uint8'), 0, 255)
        img = Image.fromarray(img_arr)

        return img

    def _dynamic(self, img, args):
        if args.dataset == 'cifar10' or args.dataset == 'CIFAR10':
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        elif args.dataset == 'cifar100' or args.dataset == 'CIFAR100':
            mean = [0.5071, 0.4867, 0.4408]
            std = [0.2675, 0.2565, 0.2761]
        elif args.dataset == 'gtsrb' or args.dataset == 'GTSRB':
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        elif args.dataset == 'ImageNet-Subset-20':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            raise Exception("Invalid dataset")

        trans_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        untrans_img = transforms.Compose([
            transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]),
            transforms.ToPILImage(),
        ])

        img_torch = trans_img(img).unsqueeze(0)
        netG = self.netG
        netM = self.netM
        patterns = netG(img_torch)
        patterns = netG.normalize_pattern(patterns)
        masks_output = netM.threshold(netM(img_torch))
        img_arr = np.array(
            untrans_img((img_torch + (patterns - img_torch) * masks_output).squeeze(0))).astype(np.uint8)
        img = Image.fromarray(img_arr)

        return img
