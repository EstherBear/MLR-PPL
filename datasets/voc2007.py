import os
import sys
sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), '..'))


import numpy as np
from PIL import Image
import xml.dom.minidom
from xml.dom.minidom import parse

import torch
import torch.utils.data as data

from config import prefixPathVOC2007

category_info = {'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4,
                 'bus':5, 'car':6, 'cat':7, 'chair':8, 'cow':9,
                 'diningtable':10, 'dog':11, 'horse':12, 'motorbike':13, 'person':14,
                 'pottedplant':15, 'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19}

class VOC2007(data.Dataset):

    def __init__(self, mode,
                 img_dir, anno_path, labels_path,
                 input_transform=None, label_proportion=1.0, positive_num=-1,
                 noise_proportion=0, noise_mode='both'):

        assert mode in ('train', 'val')

        self.mode = mode
        self.input_transform = input_transform
        self.label_proportion = label_proportion
        self.noise_proportion = noise_proportion
        self.noise_mode = noise_mode
        
        self.img_names  = []
        with open(anno_path, 'r') as f:
             self.img_names = f.readlines()
        self.img_dir = img_dir
        
        self.labels = []
        for name in self.img_names:
            label_file = os.path.join(labels_path,name[:-1]+'.xml')
            label_vector = np.zeros(20)
            DOMTree = xml.dom.minidom.parse(label_file)
            root = DOMTree.documentElement
            objects = root.getElementsByTagName('object')  
            for obj in objects:
                if (obj.getElementsByTagName('difficult')[0].firstChild.data) == '1':
                    continue
                tag = obj.getElementsByTagName('name')[0].firstChild.data.lower()
                label_vector[int(category_info[tag])] = 1.0
            self.labels.append(label_vector)

        # labels : numpy.ndarray, shape->(len(self.img_names), 20)
        # value range->(-1 means label don't exist, 1 means label exist)
        self.labels = np.array(self.labels).astype(np.int)
        self.labels[self.labels == 0] = -1

        # changedLabels : numpy.ndarray, shape->(len(self.img_names), 20)
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
        name = self.img_names[index][:-1]+'.jpg'
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

if __name__=='__main__':
    prefixPath = prefixPathVOC2007
    train_dir, train_anno, train_label = os.path.join(prefixPath, 'JPEGImages'), os.path.join(prefixPath, 'ImageSets/Main/trainval.txt'), os.path.join(prefixPath, 'Annotations')
    train_set = VOC2007('train',
                        train_dir, train_anno, train_label,
                        input_transform=None, positive_num=1)
    labels = train_set.changedLabels
    label_num = labels.sum(axis=0)
    print(label_num)