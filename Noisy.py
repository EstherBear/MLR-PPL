import os
import sys
import time
import logging
import json
from turtle import distance
import numpy as np

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler 

from model.NoisyModel import Noisy
from loss.NoisyLoss import BCELoss, AsymmetricLoss, FocalLoss, ANLoss, ANLSLoss, ROLE, ContrastiveLoss, ANFocalNegLoss, getInterPseudoLabel, getPredPseudoLabel, ANSelectLoss, LargeLoss

from utils.dataloader import  get_graph_and_word_file, get_data_loader
from utils.metrics import AverageMeter, AveragePrecisionMeter, Compute_mAP_VOC2012, \
    an_compute_inter_feature_distance, compute_pseudo_label_accuracy, compute_inter_feature_distance
from utils.checkpoint import load_pretrained_model, save_code_file, save_checkpoint
from config import arg_parse, logger, show_args

global bestPrec
bestPrec = 0

def main():
    global bestPrec
    bestEpoch = 0
    
    llr_rel = 1

    # Argument Parse
    args = arg_parse('Noisy')
    args.deltaRel /= 100
    
    # Bulid Logger
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    file_path = 'exp/log/{}.log'.format(args.post)
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Show Argument
    show_args(args)

    # Save Code File
    save_code_file(args)

    # Create dataloader
    logger.info("==> Creating dataloader...")
    train_loader, test_loader, observedLabelMatrix, groundTruthMatrix = get_data_loader(args)
    logger.info("==> Done!\n")

    # Load the network
    logger.info("==> Loading the network...")
    _, WordFile = get_graph_and_word_file(args, train_loader.dataset.changedLabels) 
    model = Noisy(WordFile, scale=args.scaleInit, bias=args.biasInit, 
                 prototypeNum=args.prototypeNum, classNum=args.classNum,
                 isAlphaLearnable=args.isAlphaLearnable, isBetaLearnable=args.isBetaLearnable,
                 useGNN=args.useGNN, usePLRB=args.usePLRB, useILRB=args.useILRB, useSemantic=args.useSemantic,
                 isCosineInit=args.isCosineInit, observedLabelMatrix=observedLabelMatrix, args=args)

    if args.pretrainedModel != 'None':
        logger.info("==> Loading pretrained model...")
        model = load_pretrained_model(model, args)

    if args.resumeModel != 'None':
        logger.info("==> Loading checkpoint...")
        checkpoint = torch.load(args.resumeModel, map_location='cpu')
        bestPrec, args.startEpoch = checkpoint['best_mAP'], checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("==> Checkpoint Epoch: {0}, mAP: {1}".format(args.startEpoch, bestPrec))

    model.cuda()
    logger.info("==> Done!\n")
    

    criterion = {'ANLoss' : ANLoss(reduce=True, size_average=True).cuda() if args.posNum != -1 else BCELoss(reduce=True, size_average=True).cuda(),
                 'ContrastiveLoss': ContrastiveLoss(args.batchSize, reduce=True, size_average=True).cuda(),
                 'BCELoss' : BCELoss(reduce=True, size_average=True).cuda(),
                 'ASL' : AsymmetricLoss(reduce=True, size_average=True).cuda(),
                 'FocalLoss' : FocalLoss(reduce=True, size_average=True).cuda(),
                 'ROLE' : ROLE(expected_num_pos=args.expectedNumPos, reduce=True, size_average=True).cuda(),
                 'FocalNegLoss' : ANFocalNegLoss(reduce=True, size_average=True, ignore_mode=args.ignoreMode, ignore_margin=args.ignoreMargin).cuda(),
                 'SelectLoss' : ANSelectLoss(select_ratio=args.selectRatio, reduce=True, size_average=True).cuda(),
                 'LargeLoss' : LargeLoss(reduce=True, size_average=True).cuda(),
                }
    
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.backbone.layer4.parameters():
        p.requires_grad = True
    
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weightDecay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepEpoch, gamma=0.1)

    if args.evaluate:
        Validate(test_loader, model, criterion, 0, args)
        return
    
    logger.info('Total: {:.3f} GB'.format(torch.cuda.get_device_properties(0).total_memory/1024.0**3))

    # Running Experiment
    logger.info("Run Experiment...")
    writer = SummaryWriter('{}/{}'.format('exp/summary/', args.post))

    predNeg = np.zeros(args.classNum)
    predPos = np.zeros(args.classNum)
    distancePos = np.zeros(args.classNum)
    distanceNeg = np.zeros(args.classNum)
    
    for epoch in range(args.startEpoch, args.startEpoch + args.epochs):

        if epoch >= args.usePLEpoch:
            if (epoch == args.usePLEpoch) or ((epoch - args.usePLEpoch) % args.recomputePrototypeInterval == 0):
                logger.info('Compute Prototype...')
                model.computePrototype(train_loader, epoch, logger)
                logger.info('Done!\n')
        predPos, predNeg, distancePos, distanceNeg = Train(
            train_loader, model, criterion, optimizer, writer, epoch, args, predPos, predNeg, distancePos, distanceNeg, llr_rel)
        llr_rel -= args.deltaRel
        if args.lossMode == 'ROLE':
            estimated = np.concatenate((model.g.get_estimated_labels(), (groundTruthMatrix>0).astype(float)), axis=1)
            estimatedmAP = Compute_mAP_VOC2012(estimated, args.classNum)
            logger.info('[Train] [Epoch {}]: mAP for estimated labels\n{}'.format(epoch, estimatedmAP))
        
        mAP = Validate(test_loader, model, criterion, epoch, args)

        # tmp
        if epoch >= args.generateLabelEpoch:
            args.interANMargin = max(0.6, args.interANMargin-0.025)

        # if epoch >= args.predLabelEpoch:
        #     args.predMargin = max(0.6, args.predMargin-0.025)
        
        # if epoch >= 0:
        #     if args.ignoreMode != 'none':
        #         args.ignoreMargin = max(0.5, args.ignoreMargin-0.1)
        #     # elif args.ignoreMode == 'random' or args.ignoreMode == 'weight':
        #     #     args.ignoreMargin = posPredAvg
        #         criterion['FocalNegLoss'] = ANFocalNegLoss(reduce=True, size_average=True, ignore_mode=args.ignoreMode, ignore_margin=args.ignoreMargin).cuda()


        scheduler.step()

        writer.add_scalar('mAP', mAP, epoch)
        torch.cuda.empty_cache()

        isBest, bestPrec = mAP > bestPrec, max(mAP, bestPrec)
        save_checkpoint(args, {'epoch':epoch, 'state_dict':model.state_dict(), 'best_mAP':mAP}, isBest)

        if isBest:
            logger.info('[Best] [Epoch {0}]: Best mAP is {1:.3f}'.format(epoch, bestPrec))
            bestEpoch = epoch
    
    logger.info('[Best] [Epoch {0}]: Best mAP is {1:.3f}'.format(bestEpoch, bestPrec))
    writer.close()

def Train(train_loader, model, criterion, optimizer, writer, epoch, args, predPos, predNeg, distancePos, distanceNeg, llr_rel):

    model.train()
    model.backbone.eval()
    model.backbone.layer4.train()

    interFeatureDistance = {'pos2pos': AverageMeter(), 'pos2neg': AverageMeter(), 'neg2neg': AverageMeter()}
    interFeatureDistanceGT = {'pos2pos': AverageMeter(), 'pos2neg': AverageMeter(), 'neg2neg': AverageMeter()}
    dropRateMeter = {'overall': [AverageMeter() for _ in range(args.classNum)], 'neg': [AverageMeter() for _ in range(args.classNum)]}
    interPseudoLabel_meter = {'accuracy': AverageMeter(), 'precision': AverageMeter(), 'recall': AverageMeter(),
                              'TP': AverageMeter(), 'TN': AverageMeter(), 'FP': AverageMeter(), 'FN': AverageMeter()}
    predPseudoLabel_meter = {'accuracy': AverageMeter(), 'precision': AverageMeter(), 'recall': AverageMeter(),
                              'TP': AverageMeter(), 'TN': AverageMeter(), 'FP': AverageMeter(), 'FN': AverageMeter()}
    category_meter = {'precision': [AverageMeter() for _ in range(args.classNum)], 'recall': [AverageMeter() for _ in range(args.classNum)]}
    category_pred_meter = {'precision': [AverageMeter() for _ in range(args.classNum)], 'recall': [AverageMeter() for _ in range(args.classNum)]}
    pred_meter = {'posPredAvg': AverageMeter(), 'negPredAvg': AverageMeter(), 'posPredAvgGT': AverageMeter(), 'negPredAvgGT': AverageMeter(), 
                  'fnPredAvg': AverageMeter(), 'fpPredAvg': AverageMeter()}
    contrastiveLoss_meter = {'pos2pos': AverageMeter(), 'pos2example': AverageMeter(), 'pos2neg': AverageMeter(), 'neg2neg': AverageMeter()}
    loss_meter = {'pos': AverageMeter(), 'neg': AverageMeter()}
    loss, loss1, loss2, loss3, loss4, loss5, loss6 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    mixedProb1, mixedProb2, positiveProb = AverageMeter(), AverageMeter(), AverageMeter()
    batch_time, data_time = AverageMeter(), AverageMeter()
    # category specific statistics
    oldPredNeg = torch.from_numpy(predNeg).cuda()
    neg_pred_category_sum = torch.zeros(args.classNum).double().cuda()
    neg_pred_category_cnt = torch.zeros(args.classNum).cuda()
    
    oldPredPos = torch.from_numpy(predPos).cuda()
    pos_pred_category_sum = torch.zeros(args.classNum).double().cuda()
    pos_pred_category_cnt = torch.zeros(args.classNum).cuda()
    
    oldDistanceNeg = torch.from_numpy(distanceNeg).double().cuda()
    neg_distance_category_sum = torch.zeros(args.classNum).cuda()
    neg_distance_category_cnt = torch.zeros(args.classNum).cuda()
    
    oldDistancePos = torch.from_numpy(distancePos).double().cuda()
    pos_distance_category_sum = torch.zeros(args.classNum).cuda()
    pos_distance_category_cnt = torch.zeros(args.classNum).cuda()
    
    TN_pred_category_sum = torch.zeros(args.classNum).double().cuda()
    TN_pred_category_cnt = torch.zeros(args.classNum).cuda()
    
    FN_pred_category_sum = torch.zeros(args.classNum).double().cuda()
    FN_pred_category_cnt = torch.zeros(args.classNum).cuda()
    
    logger.info("=========================================")

    end = time.time()
    for batchIndex, (sampleIndex, input, target, groundTruth) in enumerate(train_loader):

        input, target, groundTruth = input.float().cuda(), target.float().cuda(), groundTruth.cuda()

        # Log time of loading data
        data_time.update(time.time() - end)

        # Forward
        with torch.autograd.set_detect_anomaly(True):

            output, feature = model(input)
            
        # Update Positive Feature
        if args.useSemantic:
            model.updatePosFeature(feature, target, args.interExampleNumber, output)
            
        # Update Statistics
        negMask = (target <= 0).float()
        posMask = (target > 0).float()
        neg_pred_category_sum += torch.sum(torch.sigmoid(output.detach().clone()) * negMask, dim=0)
        neg_pred_category_cnt += torch.sum(negMask, dim=0)
        pos_pred_category_sum += torch.sum(torch.sigmoid(output.detach().clone()) * posMask, dim=0)
        pos_pred_category_cnt += torch.sum(posMask, dim=0)
        
        if args.useSemantic:
            cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-9)
            feature1 = feature.view(feature.size(0), feature.size(1), 1, feature.size(2))  # (batchSize, classNum, 1, featureDim)
            feature2 = model.posFeature                                   # (classNum, prototypeNum, featureDim)
            distance = torch.mean(cos(feature1, feature2), dim=2).detach().clone() # (batchSize, classNum)
            neg_distance_category_sum += torch.sum(distance * negMask, dim=0)
            neg_distance_category_cnt += torch.sum(negMask, dim=0)
            pos_distance_category_sum += torch.sum(distance * posMask, dim=0)
            pos_distance_category_cnt += torch.sum(posMask, dim=0)
        
        TNMask = ((target <= 0) & (groundTruth <= 0)).float()
        FNMask = ((target <= 0) & (groundTruth > 0)).float()
        TN_pred_category_sum += torch.sum(torch.sigmoid(output.detach().clone()) * TNMask, dim=0)
        TN_pred_category_cnt += torch.sum(TNMask, dim=0)
        FN_pred_category_sum += torch.sum(torch.sigmoid(output.detach().clone()) * FNMask, dim=0)
        FN_pred_category_cnt += torch.sum(FNMask, dim=0)
            # model.updatePrototypeFeature(feature, target)
            

        # Obtain PseudoLabel
        predTarget = getPredPseudoLabel(output, target, margin=args.predMargin)
        
        if args.isCategorySpecific:
            momentumDistanceNeg = batchIndex / len(train_loader) * neg_distance_category_sum / torch.clamp(neg_distance_category_cnt, min=1) + \
                (1 - batchIndex / len(train_loader)) * oldDistanceNeg
            momentumDistancePos = batchIndex / len(train_loader) * pos_distance_category_sum / torch.clamp(pos_distance_category_cnt, min=1) + \
                (1 - batchIndex / len(train_loader)) * oldDistancePos

            interCategoryMargin = torch.where(momentumDistancePos > args.interANMargin, momentumDistancePos, args.interANMargin)
            interTarget = getInterPseudoLabel(feature, target, model.posFeature, margin=args.interANMargin, 
                                            interCategoryMargin=interCategoryMargin, groundTruth=groundTruth) if args.useSemantic else target                
            
        else:
            interTarget = getInterPseudoLabel(feature, target, model.posFeature, 
                                            margin=args.interANMargin, groundTruth=groundTruth) if args.useSemantic else target
                                            # if epoch >= args.generateLabelEpoch else target
        # interTarget = groundTruth.clone()
        # interTarget[groundTruth < 0] = 0
        
        compute_pseudo_label_accuracy(predPseudoLabel_meter, category_pred_meter, predTarget, groundTruth)
        compute_pseudo_label_accuracy(interPseudoLabel_meter, category_meter, interTarget, groundTruth)

        if args.useSemantic:
            compute_inter_feature_distance(interFeatureDistance, feature, target)
            compute_inter_feature_distance(interFeatureDistanceGT, feature, groundTruth)
        
        (mixedOutput1, mixedTarget1) = model.ILRB(feature, target, output) if (epoch >= args.mixupEpoch and args.useILRB) else (None, target)
        mixedOutput2, mixedTarget2 = None, None

        mixedProb1.update(torch.sum((mixedTarget1 > 0).float()).item(), args.batchSize * args.classNum)
        positiveProb.update(torch.sum((target > 0).float()).item(), args.batchSize * args.classNum)
        
        
        pred_meter['posPredAvg'].update(torch.sum(torch.sigmoid(output[target > 0])).item(), output[target > 0].size()[0])
        pred_meter['negPredAvg'].update(torch.sum(torch.sigmoid(output[target <= 0])).item(), output[target <= 0].size()[0])
        pred_meter['posPredAvgGT'].update(torch.sum(torch.sigmoid(output[groundTruth > 0])).item(), output[groundTruth > 0].size()[0])
        pred_meter['negPredAvgGT'].update(torch.sum(torch.sigmoid(output[groundTruth <= 0])).item(), output[groundTruth <= 0].size()[0])
        pred_meter['fnPredAvg'].update(torch.sum(torch.sigmoid(output[(groundTruth > 0) & (target <= 0)])).item(), output[(groundTruth > 0) & (target <= 0)].size()[0])
        pred_meter['fpPredAvg'].update(torch.sum(torch.sigmoid(output[(groundTruth < 0) & (target > 0)])).item(), output[(groundTruth < 0) & (target > 0)].size()[0])

        if args.predNeg:
            pred = output
        else:
            pred = None
        # Compute and log loss
        if args.lossMode != 'ROLE':
            if args.lossMode == 'LargeLoss':
                loss1_ = args.clsLossWeight * criterion[args.lossMode](output, target, loss_meter, 
                                                                       llr_rel=llr_rel, dropRateMeter=dropRateMeter)
            else:
                loss1_ = args.clsLossWeight * criterion[args.lossMode](output, target, loss_meter)
        else:
            estimatedLabels = model.g(sampleIndex.cuda())
            loss1_ = args.clsLossWeight * criterion[args.lossMode](output, estimatedLabels, target)
        # cleanTarget = target.clone()
        # cleanTarget[(cleanTarget == 0) & (groundTruth == 1)] = -1
        # print(groundTruthTarget.sum() / groundTruthTarget.size(0))
        
        # if epoch >= args.generateLabelEpoch:
        #     loss2_ = (args.contrastiveLossWeight * criterion['ContrastiveLoss'](feature, interTarget, pred, model.posFeature, epoch, contrastiveLoss_meter) if epoch >= 1 else \
        #             args.contrastiveLossWeight * criterion['ContrastiveLoss'](feature, interTarget, pred, model.posFeature, epoch, contrastiveLoss_meter) * batchIndex / float(len(train_loader)))\
        #             if args.contrastiveLossWeight > 0 else 0 * loss1_
        # else:
        loss2_ = (args.contrastiveLossWeight * criterion['ContrastiveLoss'](feature, target, contrastiveLoss_meter) if epoch >= 1 else \
                args.contrastiveLossWeight * criterion['ContrastiveLoss'](feature, target, contrastiveLoss_meter) * batchIndex / float(len(train_loader)))\
                if args.contrastiveLossWeight > 0 else 0 * loss1_
        # loss2_ = (args.contrastiveLossWeight * criterion['ContrastiveLoss'](feature, groundTruthTarget, pred, model.posFeature, epoch, contrastiveLoss_meter) if epoch >= 1 else \
        #          args.contrastiveLossWeight * criterion['ContrastiveLoss'](feature, groundTruthTarget, pred, model.posFeature, epoch, contrastiveLoss_meter) * batchIndex / float(len(train_loader)))\
        #          if args.contrastiveLossWeight > 0 else 0 * loss1_
        # loss3_ = 0 * loss1_ if (epoch < args.mixupEpoch or not args.useILRB) else args.ANLossWeight * criterion['ANLoss'](mixedOutput1, mixedTarget1, loss_meter)
        
        if epoch >= args.dropEpoch:
            if args.isDropCategorySpecific:
                loss3_ = args.clsLossWeight * criterion[args.lossMode](
                    output, target, loss_meter, (momentumDistanceNeg + interCategoryMargin) * 0.5, interCategoryMargin, 
                    prototypes=model.posFeature, features=feature, dropRateMeter=dropRateMeter)
            else:
                loss3_ = args.clsLossWeight * criterion[args.lossMode](
                    output, target, loss_meter, interCategoryMargin - 0.3, interCategoryMargin, 
                    prototypes=model.posFeature, features=feature, dropRateMeter=dropRateMeter)
        else:
            loss3_ = 0 * loss1_
        # loss3_ = 0 * loss1_ if (epoch < args.mixupEpoch or not args.useILRB) else args.clsLossWeight * criterion[args.lossMode](mixedOutput1, mixedTarget1, loss_meter)
        
        loss4_ = 0 * loss1_ if epoch < args.mixupEpoch or not args.usePLRB else criterion['ANLoss'](mixedOutput2, mixedTarget2)
        # loss5_ = 0 * criterion['ANLoss'](output, interTarget) if epoch >= args.generateLabelEpoch else 0 * loss1_
        # if args.isCategorySpecific:
        #     loss5_ = args.ANLossWeight * criterion['ANLoss'](output, interTarget) if epoch >= (args.generateLabelEpoch+1) else 0 * loss1_
        # else:
        if epoch >= args.generateLabelEpoch:
                # momentumPredNeg = batchIndex / float(len(train_loader)) * neg_pred_category_sum / torch.clamp(neg_pred_category_cnt, min=1) + \
                #     (1 - batchIndex / float(len(train_loader))) * oldPredNeg
                # momentumPredPos = batchIndex / float(len(train_loader)) * pos_pred_category_sum / torch.clamp(pos_pred_category_cnt, min=1) + \
                #     (1 - batchIndex / float(len(train_loader))) * oldPredPos
            
            loss5_ = args.clsLossWeight * criterion[args.lossMode](output, interTarget, loss_meter)      
        else:
            loss5_ = 0 * loss1_
        # loss5_ = args.clsLossWeight * criterion[args.lossMode](output, interTarget, loss_meter) if epoch >= args.generateLabelEpoch else 0 * loss1_
        
        loss6_ = args.clsLossWeight * criterion[args.lossMode](output, predTarget, loss_meter) if epoch >= args.predLabelEpoch else 0 * loss1_
        loss_ = loss1_ + loss2_ + loss3_ + loss4_ + loss5_ + loss6_

        loss.update(loss_.item(), input.size(0))
        loss1.update(loss1_.item(), input.size(0))
        loss2.update(loss2_.item(), input.size(0))
        loss3.update(loss3_.item(), input.size(0))
        loss4.update(loss4_.item(), input.size(0))
        loss5.update(loss5_.item(), input.size(0))
        loss6.update(loss6_.item(), input.size(0))

        # Backward
        loss_.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log time of batch
        batch_time.update(time.time() - end)
        end = time.time()
        
        avgprecision, avgrecall, avgprecisionPred, avgrecallPred = 0, 0, 0, 0
        for classIdx in range(args.classNum):
            avgprecision += category_meter['precision'][classIdx].avg
            avgrecall += category_meter['recall'][classIdx].avg
            avgprecisionPred += category_pred_meter['precision'][classIdx].avg
            avgrecallPred += category_pred_meter['recall'][classIdx].avg
        avgprecision /= args.classNum
        avgrecall /= args.classNum
        avgprecisionPred /= args.classNum
        avgrecallPred /= args.classNum
        
        if batchIndex % args.printFreq == 0:
            logger.info('[Train] [Epoch {0}]: [{1:04d}/{2}] Batch Time {batch_time.avg:.3f} Data Time {data_time.avg:.3f} Learn Rate {lr:.6f} alpha {alpha:.4f} beta {beta:.4f}\n'
                        '\t\t\t\t\t{lossMode} {loss1.avg:.4f} Contrastive Loss {loss2.avg:.4f} Drop Loss {loss3.avg:.4f} Prototype Loss {loss4.avg:.4f}\n'
                        '\t\t\t\t\tInter AN Loss {loss5.val:.4f} ({loss5.avg:.4f}) Inter Margin {interMargin:.3f}\n'
                        '\t\t\t\t\tPred AN Loss {loss6.val:.4f} ({loss6.avg:.4f}) Pred Margin {predMargin:.3f}\n'
                        '\t\t\t\t\tInter Feature Distance (AssumeNegative): Pos2Pos {pos2pos.avg:.4f} Pos2Neg {pos2neg.avg:.4f} Neg2Neg {neg2neg.avg:.4f}\n'
                        '\t\t\t\t\tInter Feature Distance (GroundTruth): Pos2Pos {pos2posgt.avg:.4f} Pos2Neg {pos2neggt.avg:.4f} Neg2Neg {neg2neggt.avg:.4f}\n'
                        '\t\t\t\t\tInter Pseudo Label: Accuracy {accuracy.avg:.4f} Precision {precision.avg:.4f} Recall {recall.avg:.4f}\n'
                        '\t\t\t\t\t                    TP {TP.avg:.4f} TN {TN.avg:.4f} FP {FP.avg:.4f} FN {FN.avg:.4f}\n'
                        '\t\t\t\t\t                    Mean Precision {avgprecision:.4f} Mean Recall {avgrecall:.4f}\n'
                        '\t\t\t\t\tPred Pseudo Label: Accuracy {accuracyPred.avg:.4f} Precision {precisionPred.avg:.4f} Recall {recallPred.avg:.4f}\n'
                        '\t\t\t\t\t                    TP {TPPred.avg:.4f} TN {TNPred.avg:.4f} FP {FPPred.avg:.4f} FN {FNPred.avg:.4f}\n'
                        '\t\t\t\t\t                    Mean Precision {avgprecisionPred:.4f} Mean Recall {avgrecallPred:.4f}\n'
                        '\t\t\t\t\tMixup Num {mixedProb1.avg:.4f} Positive Num {positiveProb.avg:.4f}\n'
                        '\t\t\t\t\tContrastive Loss: Pos2Pos Loss {pos2posLoss.avg:.4f} Pos2Example Loss {pos2exampleLoss.avg:.4f} Pos2Neg Loss {pos2negLoss.avg:.4f} Neg2Neg Loss {neg2negLoss.avg:.4f}\n'
                        # '\t\t\t\t\tClassification Loss: Pos Loss {posLoss.avg:.4f} Neg Loss {negLoss.avg:.4f}\n'
                        '\t\t\t\t\tAverage Prediction: Pos {posPredAvg.avg:.4f} Neg {negPredAvg.avg:.4f} PosGT {posPredAvgGT.avg:.4f} NegGT {negPredAvgGT.avg:.4f} fn {fnPredAvg.avg:.4f} fp {fpPredAvg.avg:.4f}\n'
                        '\t\t\t\t\tCategory Average Prediction: Pos {posPredCatAvg:.4f} Neg {negPredCatAvg:.4f} DropRateOverall {dropRateOverall:.4f} DropRateNeg {dropRateNeg:.4f}'
                        # '\t\t\t\t\tIgnore Mode: {ignoreMode} Ignore Margin: {ignoreMargin:.3f}'
                        .format(
                        epoch, batchIndex, len(train_loader), batch_time=batch_time, data_time=data_time, lr=optimizer.param_groups[0]['lr'], alpha=model.alpha.data.item(), beta=model.beta.data.item(),
                        lossMode=args.lossMode, loss1=loss1, loss2=loss2, loss3=loss3, loss4=loss4, loss5=loss5, loss6=loss6, interMargin=args.interANMargin, predMargin=args.predMargin,
                        pos2pos=interFeatureDistance['pos2pos'], pos2neg=interFeatureDistance['pos2neg'], neg2neg=interFeatureDistance['neg2neg'], 
                        pos2posgt=interFeatureDistanceGT['pos2pos'], pos2neggt=interFeatureDistanceGT['pos2neg'], neg2neggt=interFeatureDistanceGT['neg2neg'], 
                        accuracy=interPseudoLabel_meter['accuracy'], precision=interPseudoLabel_meter['precision'], recall=interPseudoLabel_meter['recall'],
                        avgprecision=avgprecision, avgrecall=avgrecall, avgprecisionPred=avgprecisionPred, avgrecallPred=avgrecallPred,
                        TP=interPseudoLabel_meter['TP'], TN=interPseudoLabel_meter['TN'], FP=interPseudoLabel_meter['FP'], FN=interPseudoLabel_meter['FN'],
                        accuracyPred=predPseudoLabel_meter['accuracy'], precisionPred=predPseudoLabel_meter['precision'], recallPred=predPseudoLabel_meter['recall'],
                        TPPred=predPseudoLabel_meter['TP'], TNPred=predPseudoLabel_meter['TN'], FPPred=predPseudoLabel_meter['FP'], FNPred=predPseudoLabel_meter['FN'], 
                        mixedProb1=mixedProb1, positiveProb=positiveProb, pos2posLoss=contrastiveLoss_meter['pos2pos'], pos2exampleLoss=contrastiveLoss_meter['pos2example'],
                        pos2negLoss=contrastiveLoss_meter['pos2neg'], neg2negLoss=contrastiveLoss_meter['neg2neg'], ignoreMode=args.ignoreMode, ignoreMargin=args.ignoreMargin,
                        posPredAvg=pred_meter['posPredAvg'], negPredAvg=pred_meter['negPredAvg'], posPredAvgGT=pred_meter['posPredAvgGT'], 
                        negPredAvgGT=pred_meter['negPredAvgGT'], fnPredAvg=pred_meter['fnPredAvg'], fpPredAvg=pred_meter['fpPredAvg'], posLoss=loss_meter['pos'], negLoss=loss_meter['neg'], 
                        posPredCatAvg=(pos_pred_category_sum/torch.clamp(pos_pred_category_cnt, min=1)).mean(), negPredCatAvg=(neg_pred_category_sum/torch.clamp(neg_pred_category_cnt, min=1)).mean(),
                        dropRateOverall=sum([e.avg for e in dropRateMeter['overall']])/args.classNum, dropRateNeg=sum([e.avg for e in dropRateMeter['neg']])/args.classNum))
            sys.stdout.flush()
            
    # if args.isCategorySpecific:
    categoryPrecision = torch.zeros(args.classNum)
    categoryRecall = torch.zeros(args.classNum)
    categoryPrecisionPred = torch.zeros(args.classNum)
    categoryRecallPred = torch.zeros(args.classNum)
    for i in range(args.classNum):
        categoryPrecision[i] = category_meter['precision'][i].avg
        categoryRecall[i] = category_meter['recall'][i].avg
        categoryPrecisionPred[i] = category_pred_meter['precision'][i].avg
        categoryRecallPred[i] = category_pred_meter['recall'][i].avg
    
    if args.save:
        predPos = (pos_pred_category_sum / torch.clamp(pos_pred_category_cnt, min=1)).cpu().numpy()
        predNeg = (neg_pred_category_sum / torch.clamp(neg_pred_category_cnt, min=1)).cpu().numpy()
        
        distancePos = (pos_distance_category_sum / torch.clamp(pos_distance_category_cnt, min=1)).cpu().numpy() if args.useSemantic else distancePos
        distanceNeg = (neg_distance_category_sum / torch.clamp(neg_distance_category_cnt, min=1)).cpu().numpy() if args.useSemantic else distanceNeg
        predTN = (TN_pred_category_sum / torch.clamp(TN_pred_category_cnt, min=1)).cpu().numpy()
        predFN = (FN_pred_category_sum / torch.clamp(FN_pred_category_cnt, min=1)).cpu().numpy()
        
        negDropRate = np.zeros(args.classNum)
        for classIdx in range(args.classNum):
            negDropRate[classIdx] = dropRateMeter['neg'][classIdx].avg
            
        np.save('exp/statistics/{post}/{epoch}_{info}.npy'.format(post=args.post, info='PP', epoch=epoch), predPos)
        np.save('exp/statistics/{post}/{epoch}_{info}.npy'.format(post=args.post, info='PN', epoch=epoch), predNeg)
        np.save('exp/statistics/{post}/{epoch}_{info}.npy'.format(post=args.post, info='PP-PN', epoch=epoch), predPos - predNeg)
        np.save('exp/statistics/{post}/{epoch}_{info}.npy'.format(post=args.post, info='DP', epoch=epoch), distancePos)
        np.save('exp/statistics/{post}/{epoch}_{info}.npy'.format(post=args.post, info='DN', epoch=epoch), distanceNeg)
        np.save('exp/statistics/{post}/{epoch}_{info}.npy'.format(post=args.post, info='DP-DN', epoch=epoch), distancePos - distanceNeg)
        np.save('exp/statistics/{post}/{epoch}_{info}.npy'.format(post=args.post, info='PTN', epoch=epoch), predTN)
        np.save('exp/statistics/{post}/{epoch}_{info}.npy'.format(post=args.post, info='PFN', epoch=epoch), predFN)
        if epoch >= args.dropEpoch:
            np.save('exp/statistics/{post}/{epoch}_{info}.npy'.format(post=args.post, info='negDropRate', epoch=epoch), negDropRate)
    
    logger.info('[Train] [Epoch {}]: pos2posPred\n{}'.format(epoch, predPos))
    logger.info('[Train] [Epoch {}]: pos2negPred\n{}'.format(epoch, predNeg))
    logger.info('[Train] [Epoch {}]: pos2posDistance\n{}'.format(epoch, distancePos))
    logger.info('[Train] [Epoch {}]: pos2negDistance\n{}'.format(epoch, distanceNeg))
    if args.isCategorySpecific:
        logger.info('[Train] [Epoch {}]: interCategoryMargin\n{}'.format(epoch, interCategoryMargin))
    if args.isDropCategorySpecific:
        logger.info('[Train] [Epoch {}]: interCategoryMargin\n{}'.format(epoch, (momentumDistanceNeg + interCategoryMargin) * 0.5))
    logger.info('[Train] [Epoch {}]: categoryPrecision\n{}'.format(epoch, categoryPrecision))
    logger.info('[Train] [Epoch {}]: categoryRecall\n{}'.format(epoch, categoryRecall))
    # logger.info('[Train] [Epoch {}]: categoryPrecisionPred\n{}'.format(epoch, categoryPrecisionPred))
    logger.info('[Train] [Epoch {}]: categoryRecallPred\n{}'.format(epoch, categoryRecallPred))
    
    writer.add_scalar('Loss', loss.avg, epoch)
    writer.add_scalar('Loss_AN', loss1.avg, epoch)
    writer.add_scalar('Loss_Contrastive', loss2.avg, epoch)
    writer.add_scalar('Loss_Instance', loss3.avg, epoch)
    writer.add_scalar('Loss_Prototype', loss4.avg, epoch)
    writer.add_scalar('Loss_Inter_AN', loss5.avg, epoch)
    writer.add_scalar('Loss_Pred_AN', loss6.avg, epoch)
    if args.useSemantic:
        writer.add_scalar('Pos2Pos_AssumeNegative', interFeatureDistance['pos2pos'].avg, epoch)
        writer.add_scalar('Pos2Neg_AssumeNegative', interFeatureDistance['pos2neg'].avg, epoch)
        writer.add_scalar('Neg2Neg_AssumeNegative', interFeatureDistance['neg2neg'].avg, epoch)

        writer.add_scalar('Pos2Pos_GroundTruth', interFeatureDistanceGT['pos2pos'].avg, epoch)
        writer.add_scalar('Pos2Neg_GroundTruth', interFeatureDistanceGT['pos2neg'].avg, epoch)
        writer.add_scalar('Neg2Neg_GroundTruth', interFeatureDistanceGT['neg2neg'].avg, epoch)

    writer.add_scalar('Accuracy', interPseudoLabel_meter['accuracy'].avg, epoch)
    writer.add_scalar('Precision', interPseudoLabel_meter['precision'].avg, epoch)
    writer.add_scalar('Recall', interPseudoLabel_meter['recall'].avg, epoch)

    writer.add_scalar('AccuracyPred', predPseudoLabel_meter['accuracy'].avg, epoch)
    writer.add_scalar('PrecisionPred', predPseudoLabel_meter['precision'].avg, epoch)
    writer.add_scalar('RecallPred', predPseudoLabel_meter['recall'].avg, epoch)
    
    return predPos, predNeg, distancePos, distanceNeg


def Validate(val_loader, model, criterion, epoch, args):

    model.eval()

    apMeter = AveragePrecisionMeter()
    pred, loss, batch_time, data_time = [], AverageMeter(), AverageMeter(), AverageMeter()
    logger.info("=========================================")

    end = time.time()
    # inferenceTime = 0
    for batchIndex, (sampleIndex, input, target, groundTruth) in enumerate(val_loader):
        # if args.evaluate and batchIndex == 1000:
        #     logger.info(inferenceTime / 1000)
        #     break
        input, target, groundTruth = input.float().cuda(), target.float().cuda(), groundTruth.cuda()

        # Log time of loading data
        data_time.update(time.time()-end)

        # Forward
        # inferenceStart = time.time()
        with torch.no_grad():
            output = model(input)
        # inferenceTime += time.time() - inferenceStart
        # Compute loss and prediction
        loss_ = criterion['BCELoss'](output, target)
        loss.update(loss_.item(), input.size(0))

        # Change target to [0, 1]
        target[target < 0] = 0

        apMeter.add(output, target)
        pred.append(torch.cat((output, (target>0).float()), 1))

        # Log time of batch
        batch_time.update(time.time() - end)
        end = time.time()

        # logger.info information of current batch
        if batchIndex % args.printFreq == 0:
            logger.info('[Test] [Epoch {0}]: [{1:04d}/{2}] '
                  'Batch Time {batch_time.avg:.3f} Data Time {data_time.avg:.3f} '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                  epoch, batchIndex, len(val_loader),
                  batch_time=batch_time, data_time=data_time,
                  loss=loss))
            sys.stdout.flush()

    pred = torch.cat(pred, 0).cpu().clone().numpy()
    mAP = Compute_mAP_VOC2012(pred, args.classNum)

    averageAP = apMeter.value().mean()
    OP, OR, OF1, CP, CR, CF1 = apMeter.overall()
    OP_K, OR_K, OF1_K, CP_K, CR_K, CF1_K = apMeter.overall_topk(3)

    logger.info('[Test] mAP: {mAP:.3f}, averageAP: {averageAP:.3f}\n'
          '\t\t\t\t\t(Compute with all label) OP: {OP:.3f}, OR: {OR:.3f}, OF1: {OF1:.3f}, CP: {CP:.3f}, CR: {CR:.3f}, CF1:{CF1:.3f}\n'
          '\t\t\t\t\t(Compute with top-3 label) OP: {OP_K:.3f}, OR: {OR_K:.3f}, OF1: {OF1_K:.3f}, CP: {CP_K:.3f}, CR: {CR_K:.3f}, CF1: {CF1_K:.3f}\n'
          '\t\t\t\t\tinMatrix mean: {im:.3f}, inMatrix var: {iv:.3f}, outMatrix mean: {om:.3f}, outMatrix var: {ov:.3f}'.format(
          mAP=mAP, averageAP=averageAP,
          OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1, OP_K=OP_K, OR_K=OR_K, OF1_K=OF1_K, CP_K=CP_K, CR_K=CR_K, CF1_K=CF1_K, 
          im=model.inMatrix.mean(), iv=model.inMatrix.var(), om=model.outMatrix.mean(), ov=model.outMatrix.var()))

    return mAP

if __name__=="__main__":
    main()
