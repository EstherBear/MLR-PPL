import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.distributions.categorical import Categorical

from .backbone.resnet import resnet101
from .GraphNeuralNetwork import GatedGNN
from .SemanticDecoupling import SemanticDecoupling
from .Element_Wise_Layer import Element_Wise_Layer


class Noisy(nn.Module):

    def __init__(self, wordFeatures, prototypeNum, scale=1.0, bias=0.0,
                 imageFeatureDim=1024, intermediaDim=512, outputDim=1024,
                 classNum=80, wordFeatureDim=300, timeStep=3,
                 isAlphaLearnable=True, isBetaLearnable=True, useGNN=False, usePLRB=False, useILRB=False, useSemantic=False,
                 isCosineInit=False, observedLabelMatrix=None, args=None):

        super(Noisy, self).__init__()

        self.backbone = resnet101()

        self.useSemantic = useSemantic
        self.useGNN = useGNN
        self.usePLRB = usePLRB
        self.useILRB = useILRB
        self.isCosineInit = isCosineInit
        self.instanceSampleMode = args.instanceSampleMode
        self.exampleSampleMode = args.exampleSampleMode
        self.mixMode = args.mixMode
        self.args = args
        
        if imageFeatureDim != 2048:
            self.changeChannel = nn.Sequential(nn.Conv2d(2048, imageFeatureDim, kernel_size=1, stride=1, bias=False),
                                               nn.BatchNorm2d(imageFeatureDim),)

        self.classNum = classNum
        self.prototypeNum = prototypeNum

        self.timeStep = timeStep
        self.outputDim = outputDim
        self.intermediaDim = intermediaDim
        self.wordFeatureDim = wordFeatureDim
        self.imageFeatureDim = imageFeatureDim
        
        self.wordFeatures = self.load_features(wordFeatures)
        self.inMatrix, self.outMatrix = self.load_matrix(scale=scale, bias=bias)

        self.SemanticDecoupling = SemanticDecoupling(classNum, imageFeatureDim, wordFeatureDim, intermediaDim=intermediaDim)
        self.GraphNeuralNetwork = GatedGNN(imageFeatureDim, timeStep, self.inMatrix, self.outMatrix)

        self.fc = nn.Linear(2 * imageFeatureDim, outputDim)
        self.classifiers = Element_Wise_Layer(classNum, outputDim)

        # Predict from Global Featuremap
        self.fcBackbone = nn.Linear(imageFeatureDim, outputDim)
        self.classifiersBackbone = nn.Linear(imageFeatureDim, classNum) 
        self.avgPooling = nn.AdaptiveAvgPool2d((1,1))

        # Predict from Semantic Features
        self.fcSemantic = nn.Linear(imageFeatureDim, outputDim)
        self.classifiersSemantic = Element_Wise_Layer(classNum, outputDim)

        self.cos = torch.nn.CosineSimilarity(dim=3, eps=1e-9)        
        self.prototype = torch.zeros(classNum, prototypeNum, outputDim).cuda()

        self.features = [torch.zeros(10, self.outputDim) for i in range(self.classNum)]

        self.alpha = nn.Parameter(torch.tensor(0.5).float(), requires_grad=isAlphaLearnable)
        self.beta = nn.Parameter(torch.tensor(0.5).float(), requires_grad=isBetaLearnable)

        # Inter Image Correlation
        self.posFeature = None
        self.posFeaturePred = None
        # observed_label_matrix, estimated_labels, classNum
        self.g = LabelEstimator(observedLabelMatrix, estimated_labels=None, classNum=classNum)
    def forward(self, input, target=None):

        batchSize = input.size(0)

        featureMap = self.backbone(input)                                            # (batchSize, channel, imgSize, imgSize)
        if featureMap.size(1) != self.imageFeatureDim:
            featureMap = self.changeChannel(featureMap)                              # (batchSize, imgFeatureDim, imgSize, imgSize)

        if not self.useSemantic:
            feature = self.avgPooling(featureMap).squeeze()                              # (batchSize, imgFeatureDim)
            outputBackbone = torch.tanh(self.fcBackbone(feature))                                      # (batchSize, outputDim)
            resultBackbone = self.classifiersBackbone(outputBackbone)                                         # (batchSize, classNum)
            if (not self.training):
                return resultBackbone
            return resultBackbone, feature

        semanticFeature = self.SemanticDecoupling(featureMap, self.wordFeatures)     # (batchSize, classNum, outputDim)

        if not self.useGNN:
            outputSemantic = torch.tanh(self.fcSemantic(semanticFeature))                              # (batchSize, classNum, outputDim)
            resultSemantic = self.classifiersSemantic(outputSemantic)                                         # (batchSize, classNum)
            if (not self.training):
                return resultSemantic
            return resultSemantic, semanticFeature

        # Predict Category
        feature = self.GraphNeuralNetwork(semanticFeature) 
        output = torch.tanh(self.fc(torch.cat((feature.view(batchSize * self.classNum, -1),
                                               semanticFeature.view(-1, self.imageFeatureDim)), 1)))
        output = output.contiguous().view(batchSize, self.classNum, self.outputDim)
        result = self.classifiers(output)                                            # (batchSize, classNum)

        if (not self.training):
            return result

        if target is None:
            return result, semanticFeature
            
        self.alpha.data.clamp_(min=0, max=1)
        self.beta.data.clamp_(min=0, max=1)            
        
        if self.useILRB:
            # Instance-level Mixup
            coef, mixedTarget_1 = self.mixupLabel(target, torch.flip(target, dims=[0]), self.alpha)
            coef = coef.unsqueeze(-1).repeat(1, 1, self.outputDim)
            mixedSemanticFeature_1 = coef * semanticFeature + (1-coef) * torch.flip(semanticFeature, dims=[0])

            # Predict Category
            mixedFeature_1 = self.GraphNeuralNetwork(mixedSemanticFeature_1) 
            mixedOutput_1 = torch.tanh(self.fc(torch.cat((mixedFeature_1.view(batchSize * self.classNum, -1),
                                                        mixedSemanticFeature_1.view(-1, self.imageFeatureDim)), 1)))
            mixedOutput_1 = mixedOutput_1.contiguous().view(batchSize, self.classNum, self.outputDim)
            mixedResult_1 = self.classifiers(mixedOutput_1)                          # (batchSize, classNum)
        else:
            mixedResult_1, mixedTarget_1 = None, None

        if self.usePLRB:
            # Prototype-level Mixup
            prototype = self.prototype[:, torch.randint(self.prototype.size(1), (1,)), :].squeeze()
            prototype = prototype.unsqueeze(0).repeat(batchSize, 1, 1)

            mask = torch.rand(target.size()).cuda()
            mask = mask * (target == 0)
            mask[torch.arange(target.size(0)), torch.argmax(mask, dim=1)] = 1
            mask[mask != 1] = 0

            mixedSemanticFeature_2 = self.beta * semanticFeature + (1-self.beta) * prototype
            mixedSemanticFeature_2 = mask.unsqueeze(-1).repeat(1, 1, self.outputDim) * mixedSemanticFeature_2 + \
                                    (1-mask).unsqueeze(-1).repeat(1, 1, self.outputDim) * semanticFeature
            mixedTarget_2 = (1-mask) * target + mask * (1-self.beta)

            # Predict Category
            mixedFeature_2 = self.GraphNeuralNetwork(mixedSemanticFeature_2)
            mixedOutput_2 = torch.tanh(self.fc(torch.cat((mixedFeature_2.view(batchSize * self.classNum, -1),
                                                        mixedSemanticFeature_2.view(-1, self.imageFeatureDim)), 1)))
            mixedOutput_2 = mixedOutput_2.contiguous().view(batchSize, self.classNum, self.outputDim)
            mixedResult_2 = self.classifiers(mixedOutput_2)                          # (batchSize, classNum)
        else:
            mixedResult_2, mixedTarget_2 = None, None

        return result, semanticFeature, mixedResult_1, mixedTarget_1, mixedResult_2, mixedTarget_2
            
    # def mixupLabel(self, label1, label2, alpha):

    #     matrix = torch.ones_like(label1).cuda()
    #     matrix[(label1 == 0) & (label2 == 1)] = alpha

    #     return matrix, matrix * label1 + (1-matrix) * label2

    # def ILRB(self, feature, target, pred):

    #     batchSize = feature.size(0)
    #     # Instance-level Mixup
    #     if 'batch' in self.mixMode:
    #         coef, mixedTargetBatch = self.mixupLabel(target, torch.flip(target, dims=[0]), self.alpha)
    #         coef = coef.unsqueeze(-1).repeat(1, 1, self.outputDim)
    #         mixedSemanticFeatureBatch = coef * feature+ (1-coef) * torch.flip(feature, dims=[0])
    #     if 'example' in self.mixMode:
    #         if self.exampleSampleMode == 'random':
    #             example = self.posFeature[:, torch.randint(self.posFeature.size(1), (1,)), :].squeeze()
    #         elif self.exampleSampleMode == 'pred':
    #             exampleDistribution = Categorical(self.posFeaturePred)
    #             exampleIdx = exampleDistribution.sample()
    #             # print(exampleIdx)
    #             example = self.posFeature[torch.arange(self.posFeature.size(0)), exampleIdx]
    #         # print(example.size())
    #         example = example.unsqueeze(0).repeat(batchSize, 1, 1)
            
    #         if self.instanceSampleMode == 'random':
    #             mask = torch.rand(target.size()).cuda()
    #             mask = mask * (target == 0)
    #             mask[torch.arange(target.size(0)), torch.argmax(mask, dim=1)] = 1
    #         elif self.instanceSampleMode == 'pred':
    #             mask = torch.sigmoid(pred.detach().clone())
    #             # print(mask.size())
    #             # print(mask[0])    
    #             mask = mask * (target == 0)
    #             instanceDistribution = Categorical(mask)
    #             mask[torch.arange(target.size(0)), instanceDistribution.sample()] = 1

    #         mask[mask != 1] = 0
    #         # print(mask.size())
    #         # print(mask[0])
    #         # print(target.size())
    #         # print(target[0])
            
    #         mixedSemanticFeatureExample = self.alpha * feature + (1-self.alpha) * example
    #         mixedSemanticFeatureExample = mask.unsqueeze(-1).repeat(1, 1, self.outputDim) * mixedSemanticFeatureExample + \
    #                                 (1-mask).unsqueeze(-1).repeat(1, 1, self.outputDim) * feature
    #         mixedTargetExample = (1-mask) * target + mask * (1-self.alpha)

    #     if self.mixMode == 'batch':
    #         mixedSemanticFeature = mixedSemanticFeatureBatch
    #         mixedTarget = mixedTargetBatch
    #     elif self.mixMode == 'example':
    #         mixedSemanticFeature = mixedSemanticFeatureExample
    #         mixedTarget = mixedTargetExample
    #     elif self.mixMode == 'batchexampleCat':
    #         mixedSemanticFeature = torch.cat((mixedSemanticFeatureBatch, mixedSemanticFeatureExample), dim=0)
    #         mixedTarget = torch.cat((mixedTargetBatch, mixedTargetExample), dim=0)
    #     elif self.mixMode == 'batchexampleFuse':
    #         # print(mixedTargetBatch.size())
    #         # print(mixedSemanticFeatureBatch.size())
    #         mixedSemanticFeature = torch.where(mixedTargetBatch.unsqueeze(-1) > 0, mixedSemanticFeatureBatch, mixedSemanticFeatureExample)
    #         mixedTarget = torch.where(mixedTargetBatch > 0, mixedTargetBatch, mixedTargetExample)
    #     # print(mixedTarget.size())
    #     # print(mixedTarget[0])
    #     # Predict Category
    #     mixedOutputSemantic = torch.tanh(self.fcSemantic(mixedSemanticFeature))                              # (batchSize, classNum, outputDim)
    #     mixedResultSemantic = self.classifiersSemantic(mixedOutputSemantic)
    #     return mixedResultSemantic, mixedTarget
    
    def mixupLabel(self, label1, label2, alpha):
    
        matrix = torch.ones_like(label1).cuda()
        matrix[(label1 == 0) & (label2 == 1)] = alpha

        return matrix, matrix * label1 + (1-matrix) * label2

    def ILRB(self, feature, target, pred):

        batchSize = feature.size(0)
        # Instance-level Mixup
        coef, mixedTarget = self.mixupLabel(target, torch.flip(target, dims=[0]), self.alpha)
        coef = coef.unsqueeze(-1).repeat(1, 1, self.outputDim)
        mixedSemanticFeature = coef * feature+ (1-coef) * torch.flip(feature, dims=[0])
        
        mixedOutputSemantic = torch.tanh(self.fcSemantic(mixedSemanticFeature))                              # (batchSize, classNum, outputDim)
        mixedResultSemantic = self.classifiersSemantic(mixedOutputSemantic)
        return mixedResultSemantic, mixedTarget

    def computePrototype(self, train_loader, epoch, logger):
    
        from sklearn.cluster import KMeans

        self.eval()
        prototypes, features = [], [torch.zeros(10, self.outputDim) for i in range(self.classNum)]
        fpFeatures, tpFeatures = [torch.zeros(10, self.outputDim) for i in range(self.classNum)], [torch.zeros(10, self.outputDim) for i in range(self.classNum)]
        nFeatures = [torch.zeros(10, self.outputDim) for _ in range(self.classNum)]
        for batchIndex, (sampleIndex, input, target, groundTruth) in tqdm(enumerate(train_loader)):

            input, target, groundTruth = input.cuda(), target.float().cuda(), groundTruth.cuda()

            with torch.no_grad():
                featureMap = self.backbone(input)                                            # (batchSize, channel, imgSize, imgSize)
                if featureMap.size(1) != self.imageFeatureDim:
                    featureMap = self.changeChannel(featureMap)                              # (batchSize, imgFeatureDim, imgSize, imgSize)

                semanticFeature = self.SemanticDecoupling(featureMap, self.wordFeatures)     # (batchSize, classNum, outputDim)

                feature = semanticFeature.cpu()

                for i in range(self.classNum):
                    features[i] = torch.cat((features[i], feature[target[:, i] == 1, i]), dim=0)
                    fpFeatures[i] = torch.cat((fpFeatures[i], feature[(target[:, i] == 1) & (groundTruth[:, i] == -1), i]), dim=0)
                    tpFeatures[i] = torch.cat((tpFeatures[i], feature[(target[:, i] == 1) & (groundTruth[:, i] == 1), i]), dim=0)
                    nFeatures[i] = torch.cat((nFeatures[i], feature[target[:, i] == -1, i]), dim=0)
                    
        logger.info('Begin Kmeans...')
        for i in range(self.classNum):
            kmeans = KMeans(n_clusters=self.prototypeNum).fit(features[i][10:].numpy())
            prototypes.append(torch.tensor(kmeans.cluster_centers_).cuda())
        self.prototype = torch.stack(prototypes, dim=0) # (classNum, prototypeNum, outputDim)
        logger.info('Finish Kmeans...')
        fpAvgDis = torch.zeros(self.classNum) 
        tpAvgDis = torch.zeros(self.classNum) 
        fpMinDis = torch.zeros(self.classNum) 
        tpMaxDis = torch.zeros(self.classNum)
         
        pAvgDis = torch.zeros(self.classNum) 
        nAvgDis = torch.zeros(self.classNum) 
        nMinDis = torch.zeros(self.classNum) 
        for i in range(self.classNum):
            fpDis = torch.norm(fpFeatures[i].view(fpFeatures[i].size(0), 1, -1) - self.prototype[i].cpu().view(1, self.prototypeNum, -1), dim=-1).mean(dim=1)
            tpDis = torch.norm(tpFeatures[i].view(tpFeatures[i].size(0), 1, -1) - self.prototype[i].cpu().view(1, self.prototypeNum, -1), dim=-1).mean(dim=1)
            pDis = torch.norm(features[i].view(features[i].size(0), 1, -1) - self.prototype[i].cpu().view(1, self.prototypeNum, -1), dim=-1).mean(dim=1)
            nDis = torch.norm(nFeatures[i].view(nFeatures[i].size(0), 1, -1) - self.prototype[i].cpu().view(1, self.prototypeNum, -1), dim=-1).mean(dim=1)
            fpAvgDis[i] = fpDis.mean()
            fpMinDis[i] = fpDis.min()
            tpAvgDis[i] = tpDis.mean()
            tpMaxDis[i] = tpDis.max()
            
            pAvgDis[i] = pDis.mean()
            nAvgDis[i] = nDis.mean()
            nMinDis[i] = nDis.min()
        logger.info('[Train] [Epoch {}]: fpAvgDis\n{}'.format(epoch, fpAvgDis))
        logger.info(f'\t\t\t\t\ttpAvgDis\n{tpAvgDis}')
        logger.info(f'\t\t\t\t\tfpMinDis\n{fpMinDis}')
        logger.info(f'\t\t\t\t\ttpMaxDis\n{tpMaxDis}')
        
        logger.info('[Train] [Epoch {}]: pAvgDis\n{}'.format(epoch, pAvgDis))
        logger.info(f'\t\t\t\t\tnAvgDis\n{nAvgDis}')
        logger.info(f'\t\t\t\t\tmidAvgDis\n{(pAvgDis + nAvgDis)/2}')
        logger.info(f'\t\t\t\t\tnMinDis\n{nMinDis}')
        
        np.save('exp/statistics/{post}/{epoch}_{info}.npy'.format(post=self.args.post, info='fpAvgDis', epoch=epoch), fpAvgDis.numpy())
        np.save('exp/statistics/{post}/{epoch}_{info}.npy'.format(post=self.args.post, info='tpAvgDis', epoch=epoch), tpAvgDis.numpy())
        np.save('exp/statistics/{post}/{epoch}_{info}.npy'.format(post=self.args.post, info='fpMinDis', epoch=epoch), fpMinDis.numpy())
        np.save('exp/statistics/{post}/{epoch}_{info}.npy'.format(post=self.args.post, info='tpMaxDis', epoch=epoch), tpMaxDis.numpy())
        
        np.save('exp/statistics/{post}/{epoch}_{info}.npy'.format(post=self.args.post, info='pAvgDis', epoch=epoch), pAvgDis.numpy())
        np.save('exp/statistics/{post}/{epoch}_{info}.npy'.format(post=self.args.post, info='nAvgDis', epoch=epoch), nAvgDis.numpy())
        np.save('exp/statistics/{post}/{epoch}_{info}.npy'.format(post=self.args.post, info='nMinDis', epoch=epoch), nMinDis.numpy())
        np.save('exp/statistics/{post}/{epoch}_{info}.npy'.format(post=self.args.post, info='midAvgDis', epoch=epoch), (pAvgDis + nAvgDis).numpy()/2)
        
    def load_features(self, wordFeatures):
        return nn.Parameter(torch.from_numpy(wordFeatures.astype(np.float32)), requires_grad=False)

    def load_matrix(self, scale, bias):
        if self.isCosineInit:
            cos_matrix = torch.zeros(self.classNum, self.classNum)
            for label_idx in range(self.classNum):
                cos_matrix[label_idx] = F.cosine_similarity(self.wordFeatures[label_idx].unsqueeze(0), self.wordFeatures, dim=1)
            _in_matrix, _out_matrix = \
                nn.Parameter(cos_matrix, requires_grad=True), nn.Parameter(cos_matrix, requires_grad=True)
        else:
            _in_matrix, _out_matrix = \
                nn.Parameter(torch.rand(self.classNum, self.classNum)*scale + bias, requires_grad=True), \
                    nn.Parameter(torch.rand(self.classNum, self.classNum)*scale + bias, requires_grad=True)
            
        return _in_matrix, _out_matrix
    
    def updatePrototypeFeature(self, semanticFeature, target):
        feature = semanticFeature.detach().clone().cpu()
        for i in range(self.classNum):
            self.features[i] = torch.cat((self.features[i], feature[target[:, i] == 1, i]), dim=0)
    
    def updatePosFeature(self, feature, target, exampleNum, pred):
    
        if self.posFeature is None:
            self.posFeature = torch.zeros((self.classNum, exampleNum, feature.size(-1))).cuda()
            self.posFeaturePred = torch.zeros((self.classNum, exampleNum)).cuda()

        feature = feature.detach().clone()
        pred = torch.sigmoid(pred.detach().clone())
        
        for c in range(self.classNum):
            posFeature = feature[:, c][target[:, c] == 1]
            posFeaturePred = pred[:, c][target[:, c] == 1]
            self.posFeature[c] = torch.cat((posFeature, self.posFeature[c]), dim=0)[:exampleNum]
            self.posFeaturePred[c] = torch.cat((posFeaturePred, self.posFeaturePred[c]), dim=0)[:exampleNum]

def inverse_sigmoid(p):
    epsilon = 1e-5
    p = np.minimum(p, 1 - epsilon)
    p = np.maximum(p, epsilon)
    return np.log(p / (1-p))

class LabelEstimator(torch.nn.Module):
    
    def __init__(self, observed_label_matrix, estimated_labels, classNum):
        
        super(LabelEstimator, self).__init__()
        print('initializing label estimator')
        
        # Note: observed_label_matrix is assumed to have values in {-1, 0, 1} indicating 
        # observed negative, unknown, and observed positive labels, resp.
        
        num_examples = int(np.shape(observed_label_matrix)[0])
        observed_label_matrix = np.array(observed_label_matrix).astype(np.int8)
        total_pos = np.sum(observed_label_matrix == 1)
        total_neg = np.sum(observed_label_matrix == -1)
        print('observed positives: {} total, {:.1f} per example on average'.format(total_pos, total_pos / num_examples))
        print('observed negatives: {} total, {:.1f} per example on average'.format(total_neg, total_neg / num_examples))
        
        if estimated_labels is None:
            # initialize unobserved labels:
            w = 0.1
            q = inverse_sigmoid(0.5 + w)
            param_mtx = q * (2 * torch.rand(num_examples, classNum) - 1)
            
            # initialize observed positive labels:
            init_logit_pos = inverse_sigmoid(0.995)
            idx_pos = torch.from_numpy((observed_label_matrix == 1).astype(np.bool))
            param_mtx[idx_pos] = init_logit_pos
            
            # initialize observed negative labels:
            init_logit_neg = inverse_sigmoid(0.005)
            idx_neg = torch.from_numpy((observed_label_matrix == -1).astype(np.bool))
            param_mtx[idx_neg] = init_logit_neg
        else:
            param_mtx = inverse_sigmoid(torch.FloatTensor(estimated_labels))
        
        self.logits = torch.nn.Parameter(param_mtx)
        
    def get_estimated_labels(self):
        with torch.set_grad_enabled(False):
            estimated_labels = torch.sigmoid(self.logits)
        estimated_labels = estimated_labels.clone().detach().cpu().numpy()
        return estimated_labels
    
    def forward(self, indices):
        x = self.logits[indices, :]
        x = torch.sigmoid(x)
        return x
# =============================================================================
# Help Functions
# =============================================================================
