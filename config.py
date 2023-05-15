"""
Configuration file!
"""

import logging
import warnings
import argparse

warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Dataset Path
# =============================================================================
prefixPathCOCO = '/data2/dataset/MS-COCO_2014/'
prefixPathVG = '/data2/dataset/VG/'
prefixPathVOC2007 = '/data2/dataset/PASCAL/voc2007/VOCdevkit/VOC2007/'
# =============================================================================

# ClassNum of Dataset
# =============================================================================
_ClassNum = {'COCO2014': 80,
             'VOC2007': 20,
             'VG': 200,
            }
# =============================================================================

# ExpectedNumPos of Dataset
# =============================================================================
_ExpectedNumPos = {'COCO2014': 2.9,
             'VOC2007': 1.5,
             'VG': 10.7,
            }
# =============================================================================


# Argument Parse
# =============================================================================
def str2bool(input):
    if isinstance(input, bool):
        return input

    if input.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def show_args(args):

    logger.info("==========================================")
    logger.info("==========       CONFIG      =============")
    logger.info("==========================================")

    for arg, content in args.__dict__.items():
        logger.info("{}: {}".format(arg, content))

    logger.info("==========================================")
    logger.info("===========        END        ============")
    logger.info("==========================================")

    logger.info("\n")


def arg_parse(mode):

    assert mode in ('SST', 'SARB', 'SinglePositive', 'Noisy')

    parser = argparse.ArgumentParser(description='HCP Multi-label Image Recognition with Partial Labels')

    # Basic Augments
    parser.add_argument('--post', type=str, default='', help='postname of save model')
    parser.add_argument('--printFreq', type=int, default='1000', help='number of print frequency (default: 1000)')

    parser.add_argument('--mode', type=str, default='SST', choices=['SST', 'SARB', 'SinglePositive', 'Noisy'], help='mode of experiment (default: SST)')
    parser.add_argument('--dataset', type=str, default='COCO2014', choices=['COCO2014', 'VG', 'VOC2007'], help='dataset for training and testing')
    parser.add_argument('--prob', type=float, default=0.5, help='hyperparameter of label proportion (default: 0.5)')
    parser.add_argument('--posNum', type=int, default=-1, help='number of the positive label of an instance (default: -1)')
    parser.add_argument('--noise', type=float, default=0.0, help='hyperparameter of noise proportion (default: 0.0)')
    parser.add_argument('--noiseMode', type=str, default='both', choices=['pos', 'neg', 'both'], help='mode of noise (default: both)')

    parser.add_argument('--pretrainedModel', type=str, default='None', help='path to pretrained model (default: None)')
    parser.add_argument('--resumeModel', type=str, default='None', help='path to resume model (default: None)')
    parser.add_argument('--evaluate', type=str2bool, default='False', help='whether to evaluate model (default: False)')

    parser.add_argument('--epochs', type=int, default=20, help='number of total epochs to run (default: 20)')
    parser.add_argument('--startEpoch', type=int, default=0, help='manual epoch number (default: 0)')
    parser.add_argument('--stepEpoch', type=int, default=10, help='decend the lr in epoch number (default: 10)')

    parser.add_argument('--batchSize', type=int, default=8, help='mini-batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weightDecay', type=float, default=1e-4, help='weight decay (default: 0.0001)')

    parser.add_argument('--cropSize', type=int, default=448, help='size of crop image')
    parser.add_argument('--scaleSize', type=int, default=512, help='size of rescale image')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 4)')

    # Aguments for SST
    if mode == 'SST':
        parser.add_argument('--generateLabelEpoch', type=int, default=5, help='when to generate pseudo label (default: 5)')

        parser.add_argument('--intraBCEMargin', type=float, default=1.0, help='margin of intra bce loss (default: 1.0)')
        parser.add_argument('--intraBCEWeight', type=float, default=1.0, help='weight of intra bce loss (default: 1.0)')
        parser.add_argument('--intraCooccurrenceWeight', type=float, default=1.0, help='weight of intra co-occurrence loss (default: 1.0)')

        parser.add_argument('--interBCEMargin', type=float, default=1.0, help='margin of inter bce loss (default: 1.0)')
        parser.add_argument('--interBCEWeight', type=float, default=1.0, help='weight of inter bce loss (default: 1.0)')
        parser.add_argument('--interDistanceWeight', type=float, default=1.0, help='weight of inter Distance loss (default: 1.0)')
        parser.add_argument('--interExampleNumber', type=int, default=50, help='number of inter positive number (default: 50)')

    # Aguments for SARB
    if mode == 'SARB':
        parser.add_argument('--mixupEpoch', type=int, default=5, help='when to mix up (default: 5)')
        parser.add_argument('--contrastiveLossWeight', type=float, default=1.0, help='weight of contrastiveloss (default: 1.0)')
        
        parser.add_argument('--prototypeNum', type=int, default=10, help='number of the prototype (default: 10)')
        parser.add_argument('--recomputePrototypeInterval', type=int, default=5, help='interval of recomputing prototypes whose value greater than zero means recomputing prototypes (default: 5)')

        parser.add_argument('--isAlphaLearnable', type=str2bool, default='True', help='whether to set alpha be learnable  (default: True)')
        parser.add_argument('--isBetaLearnable', type=str2bool, default='True', help='whether to beta be learnable (default: True)')

    # Aguments for SinglePositive
    if mode == 'SinglePositive':
        parser.add_argument('--mixupEpoch', type=int, default=5, help='when to mix up (default: 5)')
        parser.add_argument('--contrastiveLossWeight', type=float, default=1.0, help='weight of contrastiveloss (default: 1.0)')
        
        parser.add_argument('--prototypeNum', type=int, default=10, help='number of the prototype (default: 10)')
        parser.add_argument('--recomputePrototypeInterval', type=int, default=5, help='interval of recomputing prototypes whose value greater than zero means recomputing prototypes (default: 5)')
        parser.add_argument('--prototypeMode', type=str, default='mean', choices=['mean', 'cluster', 'example'], help='mode of calculating prototype (default: mean)')

        parser.add_argument('--isAlphaLearnable', type=str2bool, default='True', help='whether to set alpha be learnable  (default: True)')
        parser.add_argument('--isBetaLearnable', type=str2bool, default='True', help='whether to beta be learnable (default: True)')

        parser.add_argument('--scaleInit', type=float, default=1.0, help='scale for the initializaion of the matrix in gnn(default: 1.0)')
        parser.add_argument('--biasInit', type=float, default=0.0, help='bias for the initializaion of the matrix in gnn(default: 0.0)')
        parser.add_argument('--isCosineInit', type=str2bool, default='True', help='whether to initialize matrix with cosine similarity (default: True)')
        parser.add_argument('--useSemantic', type=str2bool, default='True', help='whether to use semantic decoupling module (default: True)')
        parser.add_argument('--useGNN', type=str2bool, default='True', help='whether to use Gated GNN (default: True)')
        parser.add_argument('--usePLRB', type=str2bool, default='True', help='whether to use PLRB module (default: True)')
        parser.add_argument('--useILRB', type=str2bool, default='True', help='whether to use ILRB module (default: True)')
        parser.add_argument('--useLS', type=str2bool, default='False', help='whether to use label smoothing (default: False)')
        # parser.add_argument('--useASL', type=str2bool, default='False', help='whether to use asymmetric loss (default: False)')
        parser.add_argument('--lossMode', type=str, default='ANLoss', choices=['ANLoss', 'ASL', 'FocalLoss', 'ROLE', 'BCELoss', 'FocalNegLoss', 'FocalNegIgnoreLoss', 'SelectLoss'], help='mode of experiment (default: ANLoss)')
        parser.add_argument('--lsCoef', type=float, default=0.1, help='label smoothing coefficient(default: 0.1)')
        parser.add_argument('--mixMode', type=str, default='batch', choices=['batch', 'example', 'batchexampleCat', 'batchexampleFuse'], help='mode of mixup (default: batch)')
        parser.add_argument('--instanceSampleMode', type=str, default='random', choices=['random', 'pred'], help='mode of sampling instance when mixup (default: random)')
        parser.add_argument('--exampleSampleMode', type=str, default='random', choices=['random', 'pred'], help='mode of sampling example when mixup (default: random)')
        
        parser.add_argument('--generateLabelEpoch', type=int, default=5, help='when to generate pseudo label (default: 5)')
        parser.add_argument('--interANMargin', type=float, default=1.0, help='margin of inter bce loss (default: 1.0)')
        # parser.add_argument('--interANWeight', type=float, default=1.0, help='weight of inter bce loss (default: 1.0)')
        parser.add_argument('--clsLossWeight', type=float, default=1.0, help='weight of classification loss (default: 1.0)')
        parser.add_argument('--mixupWeight', type=float, default=1.0, help='weight of mixup loss (default: 1.0)')
        parser.add_argument('--interExampleNumber', type=int, default=50, help='number of inter positive number (default: 50)')
        parser.add_argument('--isCategorySpecific', type=str2bool, default='False', help='whether to use category specific margin to generate pseudo label (default: False)')
        parser.add_argument('--save', type=str2bool, default='False', help='whether to save category distance (default: False)')
        parser.add_argument('--predNeg', type=str2bool, default='False', help='whether to order negative with prediction when calculating contrastive loss (default: False)')
        # parser.add_argument('--posThresh', type=float, default=1.0, help='thresh to relabel positive when calculating contrastive loss (default: 1.0)')

        parser.add_argument('--predLabelEpoch', type=int, default=20, help='when to generate pred pseudo label (default: 20)')
        parser.add_argument('--predMargin', type=float, default=1.0, help='margin of pred an loss (default: 1.0)')
        parser.add_argument('--predWeight', type=float, default=1.0, help='weight of pred an loss (default: 1.0)')

        parser.add_argument('--ignoreMode', type=str, default='none', choices=['hard', 'random', 'weight', 'none'], help='mode of ignore (default: none)')
        parser.add_argument('--ignoreMargin', type=float, default=1.0, help='margin of ignoring (default: 1.0)')
        
        parser.add_argument('--selectRatio', type=int, default=10, help='ratio of pos sample to negative sample for selectLoss (default: 10)')
        

    # Aguments for Noisy
    if mode == 'Noisy':
        parser.add_argument('--mixupEpoch', type=int, default=5, help='when to mix up (default: 5)')
        parser.add_argument('--contrastiveLossWeight', type=float, default=1.0, help='weight of contrastiveloss (default: 1.0)')
        parser.add_argument('--dropEpoch', type=int, default=5, help='when to begin dropping (default: 5)')
        
        parser.add_argument('--prototypeNum', type=int, default=10, help='number of the prototype (default: 10)')
        parser.add_argument('--recomputePrototypeInterval', type=int, default=5, help='interval of recomputing prototypes whose value greater than zero means recomputing prototypes (default: 5)')
        parser.add_argument('--prototypeMode', type=str, default='mean', choices=['mean', 'cluster', 'example'], help='mode of calculating prototype (default: mean)')

        parser.add_argument('--isAlphaLearnable', type=str2bool, default='True', help='whether to set alpha be learnable  (default: True)')
        parser.add_argument('--isBetaLearnable', type=str2bool, default='True', help='whether to beta be learnable (default: True)')

        parser.add_argument('--scaleInit', type=float, default=1.0, help='scale for the initializaion of the matrix in gnn(default: 1.0)')
        parser.add_argument('--biasInit', type=float, default=0.0, help='bias for the initializaion of the matrix in gnn(default: 0.0)')
        parser.add_argument('--isCosineInit', type=str2bool, default='True', help='whether to initialize matrix with cosine similarity (default: True)')
        parser.add_argument('--useSemantic', type=str2bool, default='False', help='whether to use semantic decoupling module (default: False)')
        parser.add_argument('--useGNN', type=str2bool, default='False', help='whether to use Gated GNN (default: False)')
        parser.add_argument('--usePLRB', type=str2bool, default='False', help='whether to use PLRB module (default: False)')
        parser.add_argument('--useILRB', type=str2bool, default='False', help='whether to use ILRB module (default: False)')
        parser.add_argument('--useLS', type=str2bool, default='False', help='whether to use label smoothing (default: False)')
        # parser.add_argument('--useASL', type=str2bool, default='False', help='whether to use asymmetric loss (default: False)')
        parser.add_argument('--lossMode', type=str, default='ANLoss', choices=['ANLoss', 'ASL', 'FocalLoss', 'ROLE', 'BCELoss', 'FocalNegLoss', 'FocalNegIgnoreLoss', 'SelectLoss', 'LargeLoss'], help='mode of experiment (default: ANLoss)')
        parser.add_argument('--lsCoef', type=float, default=0.1, help='label smoothing coefficient(default: 0.1)')
        parser.add_argument('--mixMode', type=str, default='batch', choices=['batch', 'example', 'batchexampleCat', 'batchexampleFuse'], help='mode of mixup (default: batch)')
        parser.add_argument('--instanceSampleMode', type=str, default='random', choices=['random', 'pred'], help='mode of sampling instance when mixup (default: random)')
        parser.add_argument('--exampleSampleMode', type=str, default='random', choices=['random', 'pred'], help='mode of sampling example when mixup (default: random)')
        
        parser.add_argument('--generateLabelEpoch', type=int, default=5, help='when to generate pseudo label (default: 5)')
        parser.add_argument('--interANMargin', type=float, default=1.0, help='margin of inter bce loss (default: 1.0)')
        # parser.add_argument('--interANWeight', type=float, default=1.0, help='weight of inter bce loss (default: 1.0)')
        parser.add_argument('--ANLossWeight', type=float, default=1.0, help='weight of AN Loss (default: 1.0)')
        parser.add_argument('--clsLossWeight', type=float, default=1.0, help='weight of classification loss (default: 1.0)')
        parser.add_argument('--interExampleNumber', type=int, default=50, help='number of inter positive number (default: 50)')
        parser.add_argument('--isCategorySpecific', type=str2bool, default='False', help='whether to use category specific margin to generate pseudo label (default: False)')
        parser.add_argument('--isDropCategorySpecific', type=str2bool, default='False', help='whether to use category specific margin to drop label (default: False)')
        parser.add_argument('--save', type=str2bool, default='False', help='whether to save category distance (default: False)')
        parser.add_argument('--predNeg', type=str2bool, default='False', help='whether to order negative with prediction when calculating contrastive loss (default: False)')
        # parser.add_argument('--posThresh', type=float, default=1.0, help='thresh to relabel positive when calculating contrastive loss (default: 1.0)')

        parser.add_argument('--predLabelEpoch', type=int, default=20, help='when to generate pred pseudo label (default: 20)')
        parser.add_argument('--predMargin', type=float, default=1.0, help='margin of pred an loss (default: 1.0)')
        parser.add_argument('--predWeight', type=float, default=1.0, help='weight of pred an loss (default: 1.0)')

        parser.add_argument('--ignoreMode', type=str, default='none', choices=['hard', 'random', 'weight', 'none'], help='mode of ignore (default: none)')
        parser.add_argument('--ignoreMargin', type=float, default=1.0, help='margin of ignoring (default: 1.0)')
        parser.add_argument('--deltaRel', type=float, default=0.2, help='margin of LargeLoss (default: 0.2)')
        parser.add_argument('--selectRatio', type=int, default=10, help='ratio of pos sample to negative sample for selectLoss (default: 10)')
        
        parser.add_argument('--usePL', type=str2bool, default='False', help='whether to use PL (default: False)')
        parser.add_argument('--usePLEpoch', type=int, default=20, help='when to usePL (default: 20)')
        
    args = parser.parse_args()
    args.classNum = _ClassNum[args.dataset]    
    args.expectedNumPos = _ExpectedNumPos[args.dataset]

    return args
# =============================================================================
