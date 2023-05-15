import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def getPredPseudoLabel(pred, target, margin=1.0):
    """
    Shape of pred : (BatchSize, ClassNum)
    Shape of target : (BatchSize, ClassNum)
    """
    prob = torch.sigmoid(pred).detach().clone()
    assert(prob.size(0) == target.size(0))
    assert(prob.size(1) == target.size(1))
    assert(prob.dim() == target.dim())
    pseudolabel = target.detach().clone()
    pseudolabel[prob > margin] = 1
    return pseudolabel

def getConcatIndex(classNum):
    res = [[], []]
    for index in range(classNum-1):
        res[0] += [index for i in range(classNum-index-1)]
        res[1] += [i for i in range(index+1, classNum)]
    return res

def getInterPseudoLabel(feature, target, prototypes, margin=0.50, interCategoryMargin=None, groundTruth=None):
    """
    Shape of feature : (BatchSize, classNum, featureDim)
    Shape of target : (BatchSize, ClassNum)
    Shape of posFeature : (classNum, prototypeNum, featureDim)
    """

    batchSize, classNum, featureDim = feature.size()
    prototypeNum, pseudoLabel, cos = prototypes.size(1), target.detach().clone(), torch.nn.CosineSimilarity(dim=2, eps=1e-9)

    # you can use torch.repeat_interleave when you use pytorch >= 1.1.0
    target_ = target.clone()
    target_[target != 1] = 0
    target = target.cpu()
    groundTruth = groundTruth.cpu()
    feature1 = feature[target == -1].view(-1, 1, featureDim).repeat(1, prototypeNum, 1)  # (targetNum, prototypeNum, featureDim)
    feature2 = prototypes[np.where(target == -1)[1]]                                   # (targetNum, prototypeNum, featureDim)
    
    posDistance = torch.mean(cos(feature1, feature2), dim=1) # targetNum
    category = torch.where(target == -1)[1] # targetNum
    
    if interCategoryMargin is not None:
        posDistance[posDistance >= interCategoryMargin[category]] = 1    
        posDistance[posDistance <= interCategoryMargin[category]] = -1
    else:
        posDistance[posDistance >= margin] = 1
        posDistance[posDistance <= margin] = -1
    # # else:
    # posDistance[posDistance >= margin] = 1
    # posDistance[posDistance <= margin] = -1
        
    pseudoLabel[target == -1] = posDistance

    return pseudoLabel

LOG_EPSILON = 1e-5

class ANLoss(nn.Module):
    
    def __init__(self, reduce=None, size_average=None):
        super(ANLoss, self).__init__()
        self.reduce = reduce
        self.size_average = size_average

        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduce=False)

    def neg_log(self, x):
        return - torch.log(x + LOG_EPSILON)

    def forward(self, input, target, loss_meter):
        # input (-inf, +inf)
        # target {0, 1} (assuma target = 0 to negative)
        assert torch.min(target) >= 0
        input, target = input.float(), target.float()
        classNum = input.size(1)
        # target = (target > self.margin).float()
        # loss = self.BCEWithLogitsLoss(input, target)

        input = torch.sigmoid(input)
        # posNum = target.sum(dim=1)
        # negNum = float(classNum) - posNum
        # pos
        loss = target * self.neg_log(input)
        # loss = target * self.neg_log(input) * float(classNum)
        # loss = target * self.neg_log(input) / posNum.unsqueeze(1)
        
        # neg
        loss += (1 - target) * self.neg_log(1.0 - input)
        # loss += (1 - target) * self.neg_log(1.0 - input) / float(classNum)
        # loss += (1 - target) * self.neg_log(1.0 - input) / negNum.unsqueeze(1)
        # loss *= (classNum / 2)
        
        loss_meter['pos'].update(torch.sum(loss[target > 0.5]).item(), 1)
        loss_meter['neg'].update(torch.sum(loss[target <= 0.5]).item(), 1)
        if self.reduce:
            if self.size_average:
                return torch.mean(loss)
            return torch.sum(loss)
        return loss

class ANLSLoss(nn.Module):
    
    def __init__(self, ls_coef=0.1, reduce=None, size_average=None):
        super(ANLSLoss, self).__init__()

        self.ls_coef = ls_coef

        self.reduce = reduce
        self.size_average = size_average

        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduce=False)

    def neg_log(self, x):
        return - torch.log(x + LOG_EPSILON)

    def forward(self, input, target):
        assert torch.min(target) >= 0
        input, target = input.float(), target.float()
        input = torch.sigmoid(input)
        # pos
        loss = target * ((1.0 - self.ls_coef) * self.neg_log(input) \
                            + self.ls_coef * self.neg_log(1.0 - input))
        # print('target', target[0])
        # print('input', input[0])
        # print('loss', loss[0])
        # neg
        loss += (1 - target) * ((1.0 - self.ls_coef) * self.neg_log(1.0 - input) \
                + self.ls_coef * self.neg_log(input))
        # print('loss', loss[0])
        if self.reduce:
            if self.size_average:
                return torch.mean(loss)
            return torch.sum(loss)
        return loss

class BCELoss(nn.Module):

    def __init__(self, margin=0.0, reduce=None, size_average=None):
        super(BCELoss, self).__init__()

        self.margin = margin

        self.reduce = reduce
        self.size_average = size_average

        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduce=False)

    def forward(self, input, target, loss_meter=None, 
                negDis=None, posDis=None, prototypes=None, features=None, dropRateMeter=None):
        # input (-inf, +inf)
        # target {-1, 1}
        """
        Shape of features : (BatchSize, classNum, featureDim)
        Shape of target : (BatchSize, classNum)
        Shape of prototypes : (classNum, prototypeNum, featureDim)
        """

        input, target = input.float(), target.float()
        # input_sigmoid = torch.sigmoid(input)
        # print(dis.size())

        positive_mask = (target > self.margin).float()
        negative_mask = (target < -self.margin).float()

        positive_loss = self.BCEWithLogitsLoss(input, target)
        negative_loss = self.BCEWithLogitsLoss(-input, -target)
        
        if negDis is not None:
            batchSize, classNum, featureDim = features.size()
            prototypeNum, cos = prototypes.size(1), torch.nn.CosineSimilarity(dim=3, eps=1e-9)
            feature1 = features.view(-1, classNum, 1, featureDim).repeat(1, 1, prototypeNum, 1)  # (batchSize, classNum, prototypeNum, featureDim)
            feature2 = prototypes.view(1, classNum, -1, featureDim).repeat(batchSize, 1, 1, 1)  # (batchSize, classNum, prototypeNum, featureDim)
            dis = torch.mean(cos(feature1, feature2), dim=2) # targetNum
            delta_avg = torch.clamp((dis - negDis) / (posDis - negDis), min=0, max=1)
            drop_mask = (torch.rand_like(delta_avg) < delta_avg).float()
            negative_mask = (1 - drop_mask) * negative_mask
        
        loss = positive_mask * positive_loss + negative_mask * negative_loss
        
        if loss_meter is not None:
            loss_meter['pos'].update(torch.sum(positive_mask * positive_loss).item(), 1)
            loss_meter['neg'].update(torch.sum(negative_mask * negative_loss).item(), 1)
            
        if dropRateMeter is not None:
            for classIdx in range(classNum):
                dropRateMeter['overall'][classIdx].update(1 - (torch.sum(positive_mask[:, classIdx] + negative_mask[:, classIdx]) / target.size(0)), 1)
                dropRateMeter['neg'][classIdx].update(1 - (negative_mask[:, classIdx].sum() / torch.clamp((target[:, classIdx] < -self.margin).float().sum(), min=1)), 1)

        if self.reduce:
            if self.size_average:
                return torch.mean(loss[(positive_mask > 0) | (negative_mask > 0)]) if torch.sum(positive_mask + negative_mask) != 0 else torch.mean(loss)
            return torch.sum(loss[(positive_mask > 0) | (negative_mask > 0)]) if torch.sum(positive_mask + negative_mask) != 0 else torch.sum(loss)
        return loss

class ContrastiveLoss(nn.Module):
    """
    Document: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """

    def __init__(self, batchSize, reduce=None, size_average=None):
        super(ContrastiveLoss, self).__init__()

        self.batchSize = batchSize
        self.concatIndex = self.getConcatIndex(batchSize)

        self.reduce = reduce
        self.size_average = size_average

        self.cos = torch.nn.CosineSimilarity(dim=2, eps=1e-9)

    def forward(self, input, target, meter):
        """
        Shape of input: (BatchSize, classNum, featureDim)
        Shape of target: (BatchSize, classNum), Value range of target: (-1, 0, 1)
        """

        target_ = target.detach().clone()
        target_[target_ != 1] = 0
        pos2posTarget = target_[self.concatIndex[0]] * target_[self.concatIndex[1]]

        pos2negTarget = 1 - pos2posTarget
        pos2negTarget[(target[self.concatIndex[0]] == 0) | (target[self.concatIndex[1]] == 0)] = 0
        pos2negTarget[(target[self.concatIndex[0]] == -1) & (target[self.concatIndex[1]] == -1)] = 0

        target_ = -1 * target.detach().clone()
        target_[target_ != 1] = 0
        neg2negTarget = target_[self.concatIndex[0]] * target_[self.concatIndex[1]]

        distance = self.cos(input[self.concatIndex[0]], input[self.concatIndex[1]])

        if self.reduce:
            pos2pos_loss = (1 - distance)[pos2posTarget == 1]
            pos2neg_loss = (1 + distance)[pos2negTarget == 1]
            neg2neg_loss = (1 + distance)[neg2negTarget == 1]

            if pos2pos_loss.size(0) != 0:
                if neg2neg_loss.size(0) != 0:
                    neg2neg_loss = torch.cat((torch.index_select(neg2neg_loss, 0, torch.randperm(neg2neg_loss.size(0))[:2 * pos2pos_loss.size(0)].cuda()),
                                              torch.sort(neg2neg_loss, descending=True)[0][:pos2pos_loss.size(0)]), 0)
                if pos2neg_loss.size(0) != 0:
                    if pos2neg_loss.size(0) != 0:    
                        pos2neg_loss = torch.cat((torch.index_select(pos2neg_loss, 0, torch.randperm(pos2neg_loss.size(0))[:2 * pos2pos_loss.size(0)].cuda()),
                                                  torch.sort(pos2neg_loss, descending=True)[0][:pos2pos_loss.size(0)]), 0)

            loss = torch.cat((pos2pos_loss, pos2neg_loss, neg2neg_loss), 0)
            meter['pos2pos'].update(torch.sum(pos2pos_loss).item())
            meter['pos2neg'].update(torch.sum(pos2neg_loss).item())
            # meter['pos2example'].update(torch.sum(pos2example_loss).item())
            meter['neg2neg'].update(torch.sum(neg2neg_loss).item())

            if self.size_average:
                return torch.mean(loss) if loss.size(0) != 0 else torch.mean(torch.zeros_like(loss).cuda())
            return torch.sum(loss) if loss.size(0) != 0 else torch.sum(torch.zeros_like(loss).cuda())
 
        return distance

    def getConcatIndex(self, classNum):
        res = [[], []]
        for index in range(classNum - 1):
            res[0] += [index for i in range(classNum - index - 1)]
            res[1] += [i for i in range(index + 1, classNum)]
        return res

# class ANContrastiveLoss(nn.Module):
#     """
#     Document: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
#     """

#     def __init__(self, batchSize, reduce=None, size_average=None):
#         super(ANContrastiveLoss, self).__init__()

#         self.batchSize = batchSize
#         self.concatIndex = self.getConcatIndex(batchSize)

#         self.reduce = reduce
#         self.size_average = size_average

#         self.cos = torch.nn.CosineSimilarity(dim=2, eps=1e-9)
#         self.cosExample = torch.nn.CosineSimilarity(dim=-1, eps=1e-9)

#     def forward(self, input, target, pred, posFeature, epoch, meter):
#         """
#         Shape of input: (BatchSize, classNum, featureDim)
#         Shape of posFeature: (classNum, exampleNum, featureDim)
#         Shape of target: (BatchSize, classNum), Value range of target: (-1(ignore), 0, 1)
#         """
#         classNum, exampleNum, featureDim = input.size(1), posFeature.size(1), input.size(2)
#         target_ = target.detach().clone()
#         target_[target_ != 1] = 0
#         # if pred is not None:
#         #     assert(pred.size(0) == target.size(0))
#         #     assert(pred.size(1) == target.size(1))
#         #     assert(pred.dim() == target.dim())
#         #     target_[torch.sigmoid(pred) > pos_thresh] = 1
#         #     print("relabel use pred")
#         # batchPosFeature = input[target == 1]
#         # # # batchPosFeature = batchPosFeature.view(-1, 1, featureDim).repeat(1, exampleNum, 1)  # (targetNum, prototypeNum, featureDim)
#         # batchPosCategory = torch.where(target == 1)[1]

#         # # # batchPosCategory = torch.cat((batchPosCategory, batchPosCategory), dim=0)
#         # # # batchPosFeature = torch.cat((batchPosFeature, batchPosFeature), dim=0)

#         # exampleIdx = torch.randint(exampleNum, (batchPosCategory.size(0), ))
#         # batchPosExample = posFeature[batchPosCategory, exampleIdx]
#         # # # print('bound')
#         # # # print(batchPosExample.size())
#         # # # print(batchPosFeature.size())
#         # # # print(batchPosCategory)
       
#         # pos2example_loss = 1 - self.cosExample(batchPosFeature, batchPosExample) # positiveNum
#         # # pos2example_loss = pos2example_loss.mean(dim=1)
#         batchPosFeature = input[target == 1]
#         batchPosCategory = torch.where(target == 1)[1]
#         featureStd = torch.std(posFeature, dim=1)
#         # print(featureStd.size())
#         featureNoise = torch.randn(batchPosFeature.size(0), featureDim).cuda() * (featureStd[batchPosCategory] **0.5)
#         scale = 0.5
 
#         batchPosGen = batchPosFeature + featureNoise * scale
 
#         pos2noise_loss = 1 - self.cosExample(batchPosFeature, batchPosGen) # positiveNum
#         pos2posTarget = target_[self.concatIndex[0]] * target_[self.concatIndex[1]] # pairNum, classNum

#         pos2negTarget = 1 - pos2posTarget
#         pos2negTarget[(target[self.concatIndex[0]] == 0) & (target[self.concatIndex[1]] == 0)] = 0

#         # remove neg2neg part
#         target_ = 1 - target.detach().clone()
#         target_[target_ != 1] = 0
#         neg2negTarget = target_[self.concatIndex[0]] * target_[self.concatIndex[1]]

#         pos2negTarget[(target[self.concatIndex[0]] == -1) | (target[self.concatIndex[1]] == -1)] = 0
#         pos2posTarget[(target[self.concatIndex[0]] == -1) | (target[self.concatIndex[1]] == -1)] = 0
#         neg2negTarget[(target[self.concatIndex[0]] == -1) | (target[self.concatIndex[1]] == -1)] = 0

#         distance = self.cos(input[self.concatIndex[0]], input[self.concatIndex[1]])

#         if self.reduce:
#             pos2pos_loss = (1 - distance)[pos2posTarget == 1]
#             pos2neg_loss = (1 + distance)[pos2negTarget == 1]
#             neg2neg_loss = (1 + distance)[neg2negTarget == 1]
            
#             # pos2pos_w = 10000
#             # pos2example_w = 0.0
#             # pos2neg_w = 150
#             # neg2neg_w = 1.5
            
#             # pos2pos_w = 100
#             # # pos2example_w = 0.2
#             # pos2neg_w = 3
#             # neg2neg_w = 0.03
            
#             # pos2pos_loss *= pos2pos_w
#             # # pos2example_loss *= pos2example_w
#             # pos2neg_loss *= pos2neg_w
#             # neg2neg_loss *= neg2neg_w
            
#             # if epoch >= 1:
#             #     pos2pos_loss = torch.cat((pos2pos_loss, pos2noise_loss), 0)

#             if pos2pos_loss.size(0) != 0:
#                 if neg2neg_loss.size(0) != 0:
#                     # neg2neg_loss = torch.cat((torch.index_select(neg2neg_loss, 0, torch.randperm(neg2neg_loss.size(0))[:2 * pos2pos_loss.size(0)].cuda()),
#                     #                           torch.sort(neg2neg_loss, descending=True)[0][:pos2pos_loss.size(0)]), 0)
#                     if pred == None:
#                         neg2neg_loss = torch.index_select(neg2neg_loss, 0, torch.randperm(neg2neg_loss.size(0))[:3 * pos2pos_loss.size(0)].cuda())
#                     else:
#                         orderNeg = torch.argsort((pred[self.concatIndex[0]] + pred[self.concatIndex[1]])[neg2negTarget==1])[:1 * pos2pos_loss.size(0)]
#                         neg2neg_loss = torch.cat((torch.index_select(neg2neg_loss, 0, torch.randperm(neg2neg_loss.size(0))[:2 * pos2pos_loss.size(0)].cuda()),
#                                                   neg2neg_loss[orderNeg]), 0)
                                              
#                 if pos2neg_loss.size(0) != 0:  
#                     # pos2neg_loss = torch.cat((torch.index_select(pos2neg_loss, 0, torch.randperm(pos2neg_loss.size(0))[:2 * pos2pos_loss.size(0)].cuda()),
#                     #                           torch.sort(pos2neg_loss, descending=True)[0][:pos2pos_loss.size(0)]), 0)
#                     if pred == None:
#                         pos2neg_loss = torch.index_select(pos2neg_loss, 0, torch.randperm(pos2neg_loss.size(0))[:3 * pos2pos_loss.size(0)].cuda())
#                     else:
#                         negPred = torch.where(target[self.concatIndex[0]][pos2negTarget==1] == 0, pred[self.concatIndex[0]][pos2negTarget==1], pred[self.concatIndex[1]][pos2negTarget==1])
#                         orderNeg = torch.argsort(negPred)[:1 * pos2pos_loss.size(0)]
#                         pos2neg_loss = torch.cat((torch.index_select(pos2neg_loss, 0, torch.randperm(pos2neg_loss.size(0))[:2 * pos2pos_loss.size(0)].cuda()),
#                                                     pos2neg_loss[orderNeg]), 0)
                        
#             # else:
#             #     pos2neg_w = 0.1
#             #     neg2neg_w = 0.01
#             #     # if pred == None:
#             #     neg2neg_loss = torch.index_select(neg2neg_loss, 0, torch.randperm(neg2neg_loss.size(0))[:int(neg2neg_w * neg2neg_loss.size(0))].cuda())
#             #     pos2neg_loss = torch.index_select(pos2neg_loss, 0, torch.randperm(pos2neg_loss.size(0))[:int(pos2neg_w * pos2neg_loss.size(0))].cuda())
#                 # else:
#                 #     orderNeg2Neg = torch.argsort((pred[self.concatIndex[0]] + pred[self.concatIndex[1]])[neg2negTarget==1])[:int(neg2neg_w * neg2neg_loss.size(0) * 1/3)]
#                 #     neg2neg_loss = torch.cat((torch.index_select(neg2neg_loss, 0, torch.randperm(neg2neg_loss.size(0))[:int(neg2neg_w * neg2neg_loss.size(0) * 2/3)].cuda()),
#                 #                             neg2neg_loss[orderNeg2Neg]), 0)
#                 #     negPred = torch.where(target[self.concatIndex[0]][pos2negTarget==1] == 0, pred[self.concatIndex[0]][pos2negTarget==1], pred[self.concatIndex[1]][pos2negTarget==1])
#                 #     orderPos2Neg = torch.argsort(negPred)[:int(pos2neg_w * pos2neg_loss.size(0) * 1/3)]
#                 #     pos2neg_loss = torch.cat((torch.index_select(pos2neg_loss, 0, torch.randperm(pos2neg_loss.size(0))[:int(pos2neg_w * pos2neg_loss.size(0) * 2/3)].cuda()),
#                 #                                     pos2neg_loss[orderPos2Neg]), 0)

#             meter['pos2pos'].update(torch.sum(pos2pos_loss).item())
#             meter['pos2neg'].update(torch.sum(pos2neg_loss).item())
#             # meter['pos2example'].update(torch.sum(pos2example_loss).item())
#             meter['neg2neg'].update(torch.sum(neg2neg_loss).item())
            
#             # if epoch >= 1:
#             #     loss = torch.cat((pos2pos_loss, pos2neg_loss, neg2neg_loss, pos2example_loss), 0)
#             # else:
#             loss = torch.cat((pos2pos_loss, pos2neg_loss, neg2neg_loss), 0)
                
#             # loss = torch.cat((pos2pos_loss, pos2neg_loss), 0)
            
#             if self.size_average:
#                 return torch.mean(loss) if loss.size(0) != 0 else torch.mean(torch.zeros_like(loss).cuda())
#             return torch.sum(loss) if loss.size(0) != 0 else torch.sum(torch.zeros_like(loss).cuda())
 
#         return distance

#     def getConcatIndex(self, classNum):
#         res = [[], []]
#         for index in range(classNum - 1):
#             res[0] += [index for i in range(classNum - index - 1)]
#             res[1] += [i for i in range(index + 1, classNum)]
#         return res

class ANContrastiveLoss(nn.Module):
    """
    Document: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """

    def __init__(self, batchSize, reduce=None, size_average=None):
        super(ANContrastiveLoss, self).__init__()

        self.batchSize = batchSize
        self.concatIndex = self.getConcatIndex(batchSize)

        self.reduce = reduce
        self.size_average = size_average

        self.cos = torch.nn.CosineSimilarity(dim=2, eps=1e-9)
        self.cosExample = torch.nn.CosineSimilarity(dim=-1, eps=1e-9)

    def forward(self, input, target, pred, posFeature, epoch, meter):
        """
        Shape of input: (BatchSize, classNum, featureDim)
        Shape of posFeature: (classNum, exampleNum, featureDim)
        Shape of target: (BatchSize, classNum), Value range of target: (-1(ignore), 0, 1)
        """
        classNum, exampleNum, featureDim = input.size(1), posFeature.size(1), input.size(2)
        target_ = target.detach().clone()
        target_[target_ != 1] = 0
        if pred is not None:
            prob = torch.sigmoid(pred.detach().clone())
            
        batchPosFeature = input[target == 1]
        batchPosCategory = torch.where(target == 1)[1]
        featureStd = torch.std(posFeature, dim=1)
        # print(featureStd.size())
        featureNoise = torch.randn(batchPosFeature.size(0), featureDim).cuda() * (featureStd[batchPosCategory] **0.5)
        scale = 0.5
 
        batchPosGen = batchPosFeature + featureNoise * scale
 
        pos2noise_loss = 1 - self.cosExample(batchPosFeature, batchPosGen) # positiveNum
        pos2posTarget = target_[self.concatIndex[0]] * target_[self.concatIndex[1]] # pairNum, classNum

        pos2negTarget = 1 - pos2posTarget
        pos2negTarget[(target[self.concatIndex[0]] == 0) & (target[self.concatIndex[1]] == 0)] = 0

        # remove neg2neg part
        target_ = 1 - target.detach().clone()
        target_[target_ != 1] = 0
        neg2negTarget = target_[self.concatIndex[0]] * target_[self.concatIndex[1]]

        pos2negTarget[(target[self.concatIndex[0]] == -1) | (target[self.concatIndex[1]] == -1)] = 0
        pos2posTarget[(target[self.concatIndex[0]] == -1) | (target[self.concatIndex[1]] == -1)] = 0
        neg2negTarget[(target[self.concatIndex[0]] == -1) | (target[self.concatIndex[1]] == -1)] = 0

        distance = self.cos(input[self.concatIndex[0]], input[self.concatIndex[1]])

        if self.reduce:
            pos2pos_loss = (1 - distance)[pos2posTarget == 1]
            pos2neg_loss = (1 + distance)[pos2negTarget == 1]
            neg2neg_loss = (1 + distance)[neg2negTarget == 1]
            

            if pos2pos_loss.size(0) != 0:
                sampleNeg2NegNum = min(3 * pos2pos_loss.size(0), neg2neg_loss.size(0))
                samplePos2NegNum = min(3 * pos2pos_loss.size(0), pos2neg_loss.size(0))
            else:
                sampleNeg2NegNum = int(0.1 * neg2neg_loss.size(0))
                samplePos2NegNum = int(0.1 * pos2neg_loss.size(0))
            if neg2neg_loss.size(0) != 0:
                # neg2neg_loss = torch.cat((torch.index_select(neg2neg_loss, 0, torch.randperm(neg2neg_loss.size(0))[:2 * pos2pos_loss.size(0)].cuda()),
                #                           torch.sort(neg2neg_loss, descending=True)[0][:pos2pos_loss.size(0)]), 0)
                if pred == None:
                    neg2neg_loss = torch.index_select(neg2neg_loss, 0, torch.randperm(neg2neg_loss.size(0))[:3 * pos2pos_loss.size(0)].cuda())
                else:
                    weights = 1 - (prob[self.concatIndex[0]] + prob[self.concatIndex[1]])[neg2negTarget==1] / 2
                    sampleIdx = torch.multinomial(input=weights, num_samples=sampleNeg2NegNum, replacement=False)
                    neg2neg_loss = neg2neg_loss[sampleIdx]

                                            
            if pos2neg_loss.size(0) != 0:  
                if pred == None:
                    pos2neg_loss = torch.index_select(pos2neg_loss, 0, torch.randperm(pos2neg_loss.size(0))[:3 * pos2pos_loss.size(0)].cuda())
                else:
                    
                    weights = 1 - torch.where(target[self.concatIndex[0]][pos2negTarget==1] == 0, prob[self.concatIndex[0]][pos2negTarget==1], prob[self.concatIndex[1]][pos2negTarget==1])
                    sampleIdx = torch.multinomial(input=weights, num_samples=samplePos2NegNum, replacement=False)
                    pos2neg_loss = pos2neg_loss[sampleIdx]
                    # print(weights.size())
                    # print(sampleIdx.size())
                    # print(pos2neg_loss.size())
            

            meter['pos2pos'].update(torch.sum(pos2pos_loss).item())
            meter['pos2neg'].update(torch.sum(pos2neg_loss).item())
            # meter['pos2example'].update(torch.sum(pos2example_loss).item())
            meter['neg2neg'].update(torch.sum(neg2neg_loss).item())
            
            # if epoch >= 1:
            #     loss = torch.cat((pos2pos_loss, pos2neg_loss, neg2neg_loss, pos2example_loss), 0)
            # else:
            loss = torch.cat((pos2pos_loss, pos2neg_loss, neg2neg_loss), 0)
                
            # loss = torch.cat((pos2pos_loss, pos2neg_loss), 0)
            
            if self.size_average:
                return torch.mean(loss) if loss.size(0) != 0 else torch.mean(torch.zeros_like(loss).cuda())
            return torch.sum(loss) if loss.size(0) != 0 else torch.sum(torch.zeros_like(loss).cuda())
 
        return distance

    def getConcatIndex(self, classNum):
        res = [[], []]
        for index in range(classNum - 1):
            res[0] += [index for i in range(classNum - index - 1)]
            res[1] += [i for i in range(index + 1, classNum)]
        return res



class ANAsymmetricLoss(nn.Module):
    
    def __init__(self, margin=0.5, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, reduce=None, size_average=None):
        super(ANAsymmetricLoss, self).__init__()

        self.reduce = reduce
        self.size_average = size_average

        self.margin = margin
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, input, target, loss_meter):
        """
        Shape of input: (BatchSize, classNum)
        Shape of target: (BatchSize, classNum)
        """

        # Get positive and negative mask (assume unknown as negative)
        positive_mask = (target > self.margin).float()
        negative_mask = (target < self.margin).float()

        # Calculating Probabilities
        input_sigmoid = torch.sigmoid(input)
        input_sigmoid_pos = input_sigmoid
        input_sigmoid_neg = 1 - input_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            input_sigmoid_neg = (input_sigmoid_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        loss_pos = positive_mask * torch.log(input_sigmoid_pos.clamp(min=self.eps))
        loss_neg = negative_mask * torch.log(input_sigmoid_neg.clamp(min=self.eps))
        loss = -1 * (loss_pos + loss_neg)

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            prob = input_sigmoid_pos * positive_mask + input_sigmoid_neg * negative_mask
            one_sided_gamma = self.gamma_pos * positive_mask + self.gamma_neg * negative_mask
            one_sided_weight = torch.pow(1 - prob, one_sided_gamma)

            loss *= one_sided_weight
        loss_meter['pos'].update(torch.sum(loss[positive_mask > 0.5]).item(), 1)
        loss_meter['neg'].update(torch.sum(loss[positive_mask <= 0.5]).item(), 1)
        if self.reduce:
            if self.size_average:
                return torch.mean(loss)
            return torch.sum(loss)
        return loss
    
class AsymmetricLoss(nn.Module):
    
    def __init__(self, margin=0.0, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, reduce=None, size_average=None):
        super(AsymmetricLoss, self).__init__()

        self.reduce = reduce
        self.size_average = size_average

        self.margin = margin
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, input, target, loss_meter):
        """
        Shape of input: (BatchSize, classNum)
        Shape of target: (BatchSize, classNum)
        """

        # Get positive and negative mask (assume unknown as negative)
        positive_mask = (target > self.margin).float()
        negative_mask = (target < self.margin).float()

        # Calculating Probabilities
        input_sigmoid = torch.sigmoid(input)
        input_sigmoid_pos = input_sigmoid
        input_sigmoid_neg = 1 - input_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            input_sigmoid_neg = (input_sigmoid_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        loss_pos = positive_mask * torch.log(input_sigmoid_pos.clamp(min=self.eps))
        loss_neg = negative_mask * torch.log(input_sigmoid_neg.clamp(min=self.eps))
        loss = -1 * (loss_pos + loss_neg)

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            prob = input_sigmoid_pos * positive_mask + input_sigmoid_neg * negative_mask
            one_sided_gamma = self.gamma_pos * positive_mask + self.gamma_neg * negative_mask
            one_sided_weight = torch.pow(1 - prob, one_sided_gamma)

            loss *= one_sided_weight
        loss_meter['pos'].update(torch.sum(loss[positive_mask > 0.5]).item(), 1)
        loss_meter['neg'].update(torch.sum(loss[positive_mask <= 0.5]).item(), 1)
        if self.reduce:
            if self.size_average:
                return torch.mean(loss)
            return torch.sum(loss)
        return loss   

class ANFocalLoss(nn.Module):
    
    def __init__(self, margin=0.5, gamma=3, eps=1e-8, reduce=None, size_average=None):
        super(ANFocalLoss, self).__init__()

        self.reduce = reduce
        self.size_average = size_average

        self.margin = margin
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target, loss_meter):
        """
        Shape of input: (BatchSize, classNum)
        Shape of target: (BatchSize, classNum)
        """

        # Get positive and negative mask (assume unknown as negative)
        positive_mask = (target > self.margin).float()
        negative_mask = (target < self.margin).float()

        # Calculating Probabilities
        input_sigmoid = torch.sigmoid(input)
        input_sigmoid_pos = input_sigmoid
        input_sigmoid_neg = 1 - input_sigmoid

        # Basic CE calculation
        loss_pos = positive_mask * torch.log(input_sigmoid_pos.clamp(min=self.eps))
        loss_neg = negative_mask * torch.log(input_sigmoid_neg.clamp(min=self.eps))
        loss = -1 * (loss_pos + loss_neg)

        # Focusing
        if self.gamma > 0:
            prob = input_sigmoid_pos * positive_mask + input_sigmoid_neg * negative_mask
            one_sided_weight = torch.pow(1 - prob, self.gamma)

            loss *= one_sided_weight
        loss_meter['pos'].update(torch.sum(loss[positive_mask > 0.5]).item(), 1)
        loss_meter['neg'].update(torch.sum(loss[positive_mask <= 0.5]).item(), 1)
        if self.reduce:
            if self.size_average:
                return torch.mean(loss)
            return torch.sum(loss)
        return loss


class FocalLoss(nn.Module):
    
    def __init__(self, margin=0.0, gamma=3, eps=1e-8, reduce=None, size_average=None):
        super(FocalLoss, self).__init__()

        self.reduce = reduce
        self.size_average = size_average

        self.margin = margin
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target, loss_meter):
        """
        Shape of input: (BatchSize, classNum)
        Shape of target: (BatchSize, classNum)
        """

        # Get positive and negative mask (assume unknown as negative)
        positive_mask = (target > self.margin).float()
        negative_mask = (target < self.margin).float()

        # Calculating Probabilities
        input_sigmoid = torch.sigmoid(input)
        input_sigmoid_pos = input_sigmoid
        input_sigmoid_neg = 1 - input_sigmoid

        # Basic CE calculation
        loss_pos = positive_mask * torch.log(input_sigmoid_pos.clamp(min=self.eps))
        loss_neg = negative_mask * torch.log(input_sigmoid_neg.clamp(min=self.eps))
        loss = -1 * (loss_pos + loss_neg)

        # Focusing
        if self.gamma > 0:
            prob = input_sigmoid_pos * positive_mask + input_sigmoid_neg * negative_mask
            one_sided_weight = torch.pow(1 - prob, self.gamma)

            loss *= one_sided_weight
        loss_meter['pos'].update(torch.sum(loss[positive_mask > 0.5]).item(), 1)
        loss_meter['neg'].update(torch.sum(loss[positive_mask <= 0.5]).item(), 1)
        if self.reduce:
            if self.size_average:
                return torch.mean(loss)
            return torch.sum(loss)
        return loss

class ANFocalNegLoss(nn.Module):
    
    def __init__(self, margin=0.5, gamma=3, eps=1e-8, ignore_mode='hard', ignore_margin=1.0, reduce=None, size_average=None):
        super(ANFocalNegLoss, self).__init__()

        self.reduce = reduce
        self.size_average = size_average

        self.margin = margin
        self.gamma = gamma
        self.eps = eps

        self.ignore_mode = ignore_mode
        self.ignore_margin = ignore_margin

    def forward(self, input, target, loss_meter):
        """
        Shape of input: (BatchSize, classNum)
        Shape of target: (BatchSize, classNum)
        """

        # Calculating Probabilities
        input_sigmoid = torch.sigmoid(input)
        input_sigmoid_pos = input_sigmoid
        input_sigmoid_neg = 1 - input_sigmoid
        
        # Get positive and negative mask (assume unknown as negative)
        positive_mask = (target > self.margin).float()
        # negative_mask = ((target < self.margin) & (input_sigmoid <= self.ignore_margin)).float().clone().detach()
        # ignore_mask = ((target < self.margin) & (input_sigmoid > self.ignore_margin)).float().clone().detach()
        negative_mask = ((target < self.margin)).float()
        
        ignore_mask = None

        # Basic CE calculation
        loss_pos = positive_mask * torch.log(input_sigmoid_pos.clamp(min=self.eps))
        loss_neg = negative_mask * torch.log(input_sigmoid_neg.clamp(min=self.eps))
        loss = -1 * (loss_pos + loss_neg)

        # Focusing Neg
        if self.gamma > 0:
            prob = input_sigmoid_neg * negative_mask
            one_sided_weight = torch.pow(1 - prob, self.gamma)

            loss *= one_sided_weight

        # Ignore
        if self.ignore_mode == 'hard':
            loss *= (1 - ignore_mask)
        elif self.ignore_mode == 'weight':
            prob = (input_sigmoid_pos * ignore_mask)
            ignore_weight = torch.pow(1 - prob, self.gamma).clone().detach()
            loss *= ignore_weight
        elif self.ignore_mode == 'random':
            drop_mask = (ignore_mask * (torch.rand_like(input_sigmoid) < (input_sigmoid_pos - self.ignore_margin))).clone().detach()
            loss *= (1 - drop_mask)
        loss_meter['pos'].update(torch.sum(loss[positive_mask > 0.5]).item(), 1)
        loss_meter['neg'].update(torch.sum(loss[positive_mask <= 0.5]).item(), 1)

        if self.reduce:
            if self.size_average:
                return torch.mean(loss)
            return torch.sum(loss)
        return loss

class ANSelectLoss(nn.Module):
    
    def __init__(self, margin=0.0, eps=1e-8, select_ratio=30, reduce=None, size_average=None):
        super(ANSelectLoss, self).__init__()

        self.reduce = reduce
        self.size_average = size_average

        self.margin = margin
        self.eps = eps

        self.select_ratio=select_ratio

    def forward(self, input, target, loss_meter):
        """
        Shape of input: (BatchSize, classNum)
        Shape of target: (BatchSize, classNum)
        """

        # Calculating Probabilities
        input_sigmoid = torch.sigmoid(input)
        input_sigmoid_pos = input_sigmoid
        input_sigmoid_neg = 1 - input_sigmoid
        
        # Get positive and negative mask (assume unknown as negative)
        positive_mask = (target > self.margin)
        # negative_mask = ((target < self.margin) & (input_sigmoid <= self.ignore_margin)).float().clone().detach()
        # ignore_mask = ((target < self.margin) & (input_sigmoid > self.ignore_margin)).float().clone().detach()
        negative_mask = (target <= self.margin)
        
        # if org_target != None:
        #     pos_num = org_target[org_target == 1].size(0)
        # else:
        #     pos_num = target[target == 1].size(0)
        
        # selected CELoss
        loss_pos = input_sigmoid_pos.clamp(min=self.eps, max=1-self.eps)[positive_mask]
        loss_pos_target = target[positive_mask]
        loss_pos = loss_pos_target * torch.log(loss_pos) + (1 - loss_pos_target) * torch.log(1 - loss_pos)
        pos_num = loss_pos.size(0)
        loss_neg = torch.log(input_sigmoid_neg.clamp(min=self.eps))[negative_mask]
        weights = input_sigmoid_neg[negative_mask]
        
        try:
            selected = torch.multinomial(input=weights, num_samples=min(self.select_ratio * pos_num, loss_neg.size(0)), replacement=False)
        except:
            print(weights)
            weights = torch.nan_to_num(weights, nan=0.5)
            selected = torch.multinomial(input=weights, num_samples=min(self.select_ratio * pos_num, loss_neg.size(0)), replacement=False)
        loss_neg_target = target[negative_mask][selected]
        loss_neg = loss_neg[selected]
        loss_neg *= (1 - loss_neg_target)
        
        loss_meter['pos'].update(torch.sum(-1 * loss_pos).item(), 1)
        loss_meter['neg'].update(torch.sum(-1 * loss_neg).item(), 1)
        loss = -1 * torch.cat((loss_pos, loss_neg), dim=0)

        if self.reduce:
            if self.size_average:
                return torch.mean(loss)
            return torch.sum(loss)
        return loss


class ROLE(nn.Module):
    
    def __init__(self, expected_num_pos=2.9, reduce=None, size_average=None):
        super(ROLE, self).__init__()

        self.reduce = reduce
        self.size_average = size_average
        self.expected_num_pos = expected_num_pos

    def forward(self, input, estimated_labels, target):
        """
        Shape of input: (BatchSize, classNum)
        Shape of target: (BatchSize, classNum)
        """
        classNum = target.size(1)
        
        # Calculating Probabilities
        input_sigmoid = torch.sigmoid(input)
        
        # (image classifier) compute loss w.r.t. observed positives:
        loss_mtx_pos_1 = torch.zeros_like(target)
        loss_mtx_pos_1[target == 1] = self.neg_log(input_sigmoid[target == 1])
        
        # (image classifier) compute loss w.r.t. label estimator outputs:
        estimated_labels_detached = estimated_labels.detach()
        loss_mtx_cross_1 = estimated_labels_detached * self.neg_log(input_sigmoid) + (1.0 - estimated_labels_detached) * self.neg_log(1.0 - input_sigmoid)
        
        # (image classifier) compute regularizer: 
        reg_1 = self.expected_positive_regularizer(input_sigmoid, norm='2') / (classNum ** 2)
        
        # (label estimator) compute loss w.r.t. observed positives:
        loss_mtx_pos_2 = torch.zeros_like(target)
        loss_mtx_pos_2[target == 1] = self.neg_log(estimated_labels[target == 1])
        
        # (label estimator) compute loss w.r.t. image classifier outputs:
        input_sigmoid_detached = input_sigmoid.detach()
        loss_mtx_cross_2 = input_sigmoid_detached * self.neg_log(estimated_labels) + (1.0 - input_sigmoid_detached) * self.neg_log(1.0 - estimated_labels)

        # (label estimator) compute regularizer:
        reg_2 = self.expected_positive_regularizer(estimated_labels, norm='2') / (classNum ** 2)
        
        # compute final loss matrix:
        reg_loss = 0.5 * (reg_1 + reg_2)
        loss_mtx = 0.5 * (loss_mtx_pos_1 + loss_mtx_pos_2)
        loss_mtx += 0.5 * (loss_mtx_cross_1 + loss_mtx_cross_2)
        
        loss = loss_mtx + reg_loss
        
        if self.reduce:
            if self.size_average:
                return torch.mean(loss)
            return torch.sum(loss)
        return loss

    def neg_log(self, x):
        return - torch.log(x + LOG_EPSILON)
    
    def expected_positive_regularizer(self, preds, norm='2'):
        # Assumes predictions in [0,1].
        if norm == '1':
            reg = torch.abs(preds.sum(1).mean(0) - self.expected_num_pos)
        elif norm == '2':
            reg = (preds.sum(1).mean(0) - self.expected_num_pos)**2
        else:
            raise NotImplementedError
        return reg
    

class LargeLoss(nn.Module):
    
    def __init__(self, margin=0.0, reduce=None, size_average=None):
        super(LargeLoss, self).__init__()

        self.margin = margin

        self.reduce = reduce
        self.size_average = size_average

        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduce=False)

    def forward(self, input, target, loss_meter=None, llr_rel=None, dropRateMeter=None):
        # input (-inf, +inf)
        # target {-1, 1}
        """
        Shape of features : (BatchSize, classNum, featureDim)
        Shape of target : (BatchSize, classNum)
        Shape of prototypes : (classNum, prototypeNum, featureDim)
        """

        input, target = input.float(), target.float()
        batchSize, classNum = target.size()
        
        # input_sigmoid = torch.sigmoid(input)
        # print(dis.size())

        positive_mask = (target > self.margin).float()
        negative_mask = (target < -self.margin).float()

        positive_loss = self.BCEWithLogitsLoss(input, target)
        negative_loss = self.BCEWithLogitsLoss(-input, -target)
        
        unobserved_loss = negative_mask * negative_loss
        k = math.ceil(batchSize * classNum * (1-llr_rel))
        if k != 0:
            topk = torch.topk(unobserved_loss.flatten(), k)
            topk_lossvalue = topk.values[-1]
            loss_mask = (unobserved_loss < topk_lossvalue).float() 
            negative_mask *= loss_mask
        loss = positive_mask * positive_loss + negative_mask * negative_loss
        
        if loss_meter is not None:
            loss_meter['pos'].update(torch.sum(positive_mask * positive_loss).item(), 1)
            loss_meter['neg'].update(torch.sum(negative_mask * negative_loss).item(), 1)
            
        if dropRateMeter is not None:
            for classIdx in range(classNum):
                dropRateMeter['overall'][classIdx].update(1 - (torch.sum(positive_mask[:, classIdx] + negative_mask[:, classIdx]) / target.size(0)), 1)
                dropRateMeter['neg'][classIdx].update(1 - (negative_mask[:, classIdx].sum() / torch.clamp((target[:, classIdx] < -self.margin).float().sum(), min=1)), 1)

        if self.reduce:
            if self.size_average:
                return torch.mean(loss[(positive_mask > 0) | (negative_mask > 0)]) if torch.sum(positive_mask + negative_mask) != 0 else torch.mean(loss)
            return torch.sum(loss[(positive_mask > 0) | (negative_mask > 0)]) if torch.sum(positive_mask + negative_mask) != 0 else torch.sum(loss)
        return loss