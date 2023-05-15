import os
import sys
import json
import random
import numpy as np
from PIL import Image

sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.utils.data as data
import torchvision.datasets as datasets

from config import prefixPathVG

class VG(data.Dataset):

    def __init__(self, mode,
                 image_dir, anno_path, labels_path,
                 input_transform=None, label_proportion=1.0, positive_num=-1, 
                 noise_proportion=0, noise_mode='both'):

        assert mode in ('train', 'val')

        self.mode = mode
        self.input_transform = input_transform
        self.label_proportion = label_proportion
        self.noise_proportion = noise_proportion
        self.noise_mode = noise_mode

        self.img_dir = image_dir
        self.imgName_path = anno_path
        self.img_names = open(self.imgName_path, 'r').readlines()

        # labels : numpy.ndarray, shape->(len(vg), 200)
        # value range->(-1 means label don't exist, 1 means label exist)
        self.labels_path = labels_path
        _ = json.load(open(self.labels_path, 'r'))
        self.labels = np.zeros((len(self.img_names), 200)).astype(np.int) - 1
        for i in range(len(self.img_names)):
            self.labels[i][_[self.img_names[i][:-1]]] = 1

        # changedLabels : numpy.ndarray, shape->(len(vg), 200)
        # value range->(-1 means label don't exist, 0 means not sure whether the label exists, 1 means label exist)
        self.changedLabels = self.labels
        if label_proportion != 1:
            print('Changing label proportion...')
            self.changedLabels = changeLabelProportion(self.labels, self.label_proportion)

        if positive_num != -1:
            print('Ensure {} positive for each instance...'.format(positive_num))
            self.changedLabels = getNPositiveLabel(self.labels, positive_num)
            # print(self.changedLabels.dtype)
            # remove the instance without positive label
            mask = np.any(self.changedLabels, axis=1)
            self.changedLabels = self.changedLabels[mask]
            self.labels = self.labels[mask]
            self.img_names = np.array(self.img_names)[mask].tolist()
            # print(self.changedLabels.sum())
            # print(self.ids.shape)
            # print(self.changedLabels.sum())
            # print(self.changedLabels.sum() / self.changedLabels.shape[0])
        if noise_proportion != 0:
            print('Changing label with noise...')
            self.changedLabels = changeLabelNoise(self.labels, self.noise_proportion, self.noise_mode)
        print(np.where(self.changedLabels == 1)[0].shape[0])
        print(np.where(self.changedLabels == -1)[0].shape[0])
        print(np.where(self.changedLabels == 1)[0].shape[0] / self.changedLabels.shape[0])
        print(np.where(self.changedLabels == -1)[0].shape[0] / self.changedLabels.shape[0])
        print(self.changedLabels.shape[0])

    def __getitem__(self, index):
        name = self.img_names[index][:-1]
        input = Image.open(os.path.join(self.img_dir, name)).convert('RGB')
        if self.input_transform:
           input = self.input_transform(input)
        return index, input, self.changedLabels[index], self.labels[index]

    def __len__(self):
        return len(self.img_names)

# =============================================================================
# Help Functions
# =============================================================================
def getNPositiveLabel(org_labels, N):
    
    # Set Random Seed
    np.random.seed(0)

    num_items, num_classes = np.shape(org_labels)
    labels = np.zeros_like(org_labels)
    for i in range(num_items):
        idx_all = np.nonzero(org_labels[i] == 1)[0]
        np.random.shuffle(idx_all)
        labels[i, idx_all[:N]] = 1.0
    return labels


def changeLabelProportion(labels, label_proportion):

    # Set Random Seed
    np.random.seed(0)

    mask = np.random.random(labels.shape)
    mask[mask < label_proportion] = 1
    mask[mask < 1] = 0
    label = mask * labels

    assert label.shape == labels.shape

    return label

def changeLabelNoise(labels, noise_proportion, noise_mode):
    
    # Set Random Seed
    np.random.seed(0)
    
    noiseMask = np.random.random(labels.shape)
    if noise_mode == 'pos':
        noiseMask[(noiseMask < noise_proportion) & (labels == 1)] = 1
    elif noise_mode == 'neg':
        noiseMask[(noiseMask < noise_proportion) & (labels == -1)] = 1
    else:
        noiseMask[noiseMask < noise_proportion] = 1
        
    noiseMask[noiseMask < 1] = 0
    label = np.where(noiseMask == 1, -labels, labels)
    
    assert label.shape == labels.shape

    return label

def getPairIndexes(labels):

    res = []
    for index in range(labels.shape[0]):
        tmp = []
        for i in range(labels.shape[1]):
            if labels[index, i] > 0:
                tmp += np.where(labels[:, i] > 0)[0].tolist()

        tmp = set(tmp)
        tmp.discard(index)
        res.append(np.array(list(tmp)))

    return res


if __name__=='__main__':
    prefixPath = prefixPathVG
    train_dir, train_anno, train_label = os.path.join(prefixPath, 'VG_100K'), './data/vg/train_list_500.txt', './data/vg/vg_category_200_labels_index.json'
    train_set = VG('train',
                        train_dir, train_anno, train_label,
                        input_transform=None, positive_num=1)
    labels = train_set.changedLabels
    label_num = labels.sum(axis=0)
    print(label_num)

