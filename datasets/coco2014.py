import os
import sys
sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), '..', 'cocoapi/PythonAPI'))
sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.datasets as datasets

from pycocotools.coco import COCO
from config import prefixPathCOCO


class COCO2014(data.Dataset):

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

        self.root = image_dir
        self.coco = COCO(anno_path)
        self.ids = list(self.coco.imgs.keys())
     
        with open('./data/coco/category.json','r') as load_category:
            self.category_map = json.load(load_category)

        # labels : numpy.ndarray, shape->(len(coco), 80)
        # value range->(-1 means label don't exist, 1 means label exist)
        self.labels = []
        for i in range(len(self.ids)):
            img_id = self.ids[i]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)
            self.labels.append(getLabelVector(getCategoryList(target), self.category_map))
        self.labels = np.array(self.labels)
        self.labels[self.labels == 0] = -1

        # changedLabels : numpy.ndarray, shape->(len(coco), 80)
        # value range->(-1 means label don't exist, 0 means not sure whether the label exists, 1 means label exist)
        self.changedLabels = self.labels
        if label_proportion != 1:
            print('Changing label proportion...')
            self.changedLabels = changeLabelProportion(self.labels, self.label_proportion)
            # print(self.changedLabels.dtype)
        if positive_num != -1:
            print('Ensure {} positive for each instance...'.format(positive_num))
            self.changedLabels = getNPositiveLabel(self.labels, positive_num)
            # print(self.changedLabels.dtype)
            # remove the instance without positive label
            mask = np.any(self.changedLabels, axis=1)
            self.changedLabels = self.changedLabels[mask]
            self.labels = self.labels[mask]
            self.ids = np.array(self.ids)[mask].tolist()
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
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        input = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input)
        return index, input, self.changedLabels[index], self.labels[index]

    def __len__(self):
        return len(self.ids)

# =============================================================================
# Help Functions
# =============================================================================
def getCategoryList(item):
    categories = set()
    for t in item:
        categories.add(t['category_id'])
    return list(categories)


def getLabelVector(categories, category_map):
    label = np.zeros(80)
    for c in categories:
        label[category_map[str(c)]-1] = 1.0
    return label


def getLabel(mode):

    assert mode in ('train', 'val')

    from utils.dataloader import get_data_path
    train_dir, train_anno, train_label, \
    test_dir, test_anno, test_label = get_data_path('COCO2014')

    if mode == 'train':
        image_dir, anno_path = train_dir, train_anno
    else:
        image_dir, anno_path = test_dir, test_anno

    coco = datasets.CocoDetection(root=image_dir, annFile=anno_path)
    with open('./data/coco/category.json', 'r') as load_category:
        category_map = json.load(load_category)

    labels = []
    for i in range(len(coco)):
        labels.append(getLabelVector(getCategoryList(coco[i][1]), category_map))
    labels = np.array(labels).astype(np.float64)

    np.save('./data/coco/{}_label_vectors.npy'.format(mode), labels)


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


def getCoOccurrenceLabel(mode):

    assert mode in ('train', 'val')

    if mode == 'train':
        label_path = './data/coco/train_label_vectors.npy'
    else:
        label_path = './data/coco/val_label_vectors.npy'

    labels = np.load(label_path).astype(np.float64)

    coOccurrenceLabel = np.zeros((labels.shape[0], sum([i for i in range(80)])), dtype=np.float64)
    for index in range(labels.shape[0]):
        correlationMatrix = labels[index][:, np.newaxis] * labels[index][np.newaxis, :]

        index_ = 0
        for i in range(80):
            for j in range(i + 1, 80):
                if correlationMatrix[i, j] > 0:
                    coOccurrenceLabel[index, index_] = 1
                index_ += 1

    np.save('./data/coco/{}_co-occurrence_label_vectors.npy'.format(mode), coOccurrenceLabel)


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
    prefixPath = prefixPathCOCO
    train_dir, train_anno, train_label = os.path.join(prefixPath, 'train2014'), os.path.join(prefixPath, 'annotations/instances_train2014.json'), './data/coco/train_label_vectors.npy'
    train_set = COCO2014('train',
                        train_dir, train_anno, train_label,
                        input_transform=None, noise_proportion=0.2, noise_mode='pos')
    labels = train_set.changedLabels
    label_num = labels.sum(axis=0)
    print(label_num)