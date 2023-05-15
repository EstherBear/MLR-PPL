import os
import gc
import shutil

import torch


def load_pretrained_model(model, args):
    
    modelDict = model.backbone.state_dict()
    
    pretrainedModel = torch.load(args.pretrainedModel)
    pretrainedDict = {}
    for k,v in pretrainedModel.items():
        if k.startswith('fc'):
            continue
        pretrainedDict[k] = v
    modelDict.update(pretrainedDict)
    model.backbone.load_state_dict(modelDict)

    del pretrainedModel
    del pretrainedDict
    gc.collect()

    return model


def save_code_file(args):

    statisticsPath = os.path.join('exp/statistics', args.post)
    if not os.path.exists(statisticsPath):
        os.mkdir(statisticsPath)

    prefixPath = os.path.join('exp/code/', args.post)
    if not os.path.exists(prefixPath):
        os.mkdir(prefixPath)

    fileNames = []
    if args.mode == 'SST':
        fileNames = ['scripts/SST.sh', 'SST.py', 'model/SST.py', 'loss/SST.py', 'config.py']

    if args.mode == 'SinglePositive':
        fileNames = ['scripts/SinglePositive.sh', 'SinglePositive.py', 'model/SinglePositive.py', 'loss/SinglePositive.py', 'config.py']
    if args.mode == 'Noisy':
        fileNames = ['scripts/Noisy.sh', 'Noisy.py', 'model/NoisyModel.py', 'loss/NoisyLoss.py', 'config.py']
    for fileName in fileNames:
        shutil.copyfile(fileName, os.path.join(prefixPath, fileName.split('/')[-1]))


def save_checkpoint(args, state, isBest):

    outputPath = os.path.join('exp/checkpoint/', args.post)
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    torch.save(state, os.path.join(outputPath, 'Checkpoint_Current.pth'))
    if isBest:
        torch.save(state, os.path.join(outputPath, 'Checkpoint_Best.pth'))
