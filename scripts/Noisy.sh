#!/bin/bash

# Make for COCOAPI
# cd cocoapi/PythonAPI
# make -j8
# cd ../..

post='0515_Noisy0.5pos_COCO_Semantic_BCELoss_Con0.05_CST_Category_Drop0.5_0'
# post='test'

printFreq=1000 # check
mode='Noisy'
dataset='COCO2014' # check 

pretrainedModel='./data/checkpoint/resnet101.pth'
resumeModel='None'
# resumeModel='./exp/checkpoint/0329_SinglePositive_COCO_SSGRL_AN10/Checkpoint_Current.pth'
evaluate='False'
epochs=20
startEpoch=0
stepEpoch=10
workers=8

batchSize=8
lr=1e-5
momentum=0.9
weightDecay=5e-4

cropSize=448
scaleSize=512

prob=1.0
posNum=-1 # check
noise=0.5 # check
noiseMode='pos' # check

scaleInit=1.0
biasInit=0.0
isCosineInit='True'

lossMode='BCELoss' # check
useLS='False' 
lsCoef=0.1

# COCO
# BCELoss 10
# ANLoss 10
# ASL 30
# FocalLoss 60
# ROLE 10
# SelectLoss 5

# VG
# BCELoss 10
# ANLoss 10
# ASL 30
# FocalLoss 60
# ROLE 10
# SelectLoss 2

# VOC
# BCELoss 10
# ANLoss 10
# ASL 30
# FocalLoss 60
# ROLE 10
# SelectLoss 8
clsLossWeight=1 # check
contrastiveLossWeight=0.05  # 0.05 # check
# 0.2 ASL
# 0.3 FocalLoss, ANLoss
useGNN='False' # check
useSemantic='True' # check

mixupEpoch=20
usePLRB='False'
useILRB='False'
instanceSampleMode='random'
exampleSampleMode='random'
mixMode='batch'

usePLEpoch=20 # check
prototypeNum=10
prototypeMode='example'
recomputePrototypeInterval=5

isAlphaLearnable='False'
isBetaLearnable='True'

generateLabelEpoch=5 # check
interANMargin=0.95 # check
interExampleNumber=100
isCategorySpecific='True' # check
isDropCategorySpecific='True' # check

predLabelEpoch=20 # check
predMargin=1.0 # check

dropEpoch=0 # check

predNeg='True' 
save='True'

ignoreMode='none'
ignoreMargin=1.0

selectRatio=10

deltaRel=0.1 # check

# use single gpu (eg,gpu 0) to trian:
#     CUDA_VISIBLE_DEVICES=0 
# use multiple gpu (eg,gpu 0 and 1) to train
#     CUDA_VISIBLE_DEVICES=0,1  
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=4 python Noisy.py \
    --post ${post} \
    --printFreq ${printFreq} \
    --mode ${mode} \
    --dataset ${dataset} \
    --prob ${prob} \
    --pretrainedModel ${pretrainedModel} \
    --resumeModel ${resumeModel} \
    --evaluate ${evaluate} \
    --epochs ${epochs} \
    --startEpoch ${startEpoch} \
    --stepEpoch ${stepEpoch} \
    --batchSize ${batchSize} \
    --lr ${lr} \
    --momentum ${momentum} \
    --weightDecay ${weightDecay} \
    --cropSize ${cropSize} \
    --scaleSize ${scaleSize} \
    --workers ${workers} \
    --mixupEpoch ${mixupEpoch} \
    --contrastiveLossWeight ${contrastiveLossWeight} \
    --prototypeNum ${prototypeNum} \
    --recomputePrototypeInterval ${recomputePrototypeInterval} \
    --isAlphaLearnable ${isAlphaLearnable} \
    --isBetaLearnable ${isBetaLearnable} \
    --posNum ${posNum} \
    --useGNN ${useGNN} \
    --usePLRB ${usePLRB} \
    --useILRB ${useILRB} \
    --useSemantic ${useSemantic} \
    --lsCoef ${lsCoef} \
    --useLS ${useLS} \
    --generateLabelEpoch ${generateLabelEpoch} \
    --interANMargin ${interANMargin} \
    --interExampleNumber ${interExampleNumber} \
    --isCosineInit ${isCosineInit} \
    --predLabelEpoch ${predLabelEpoch} \
    --predMargin ${predMargin} \
    --predNeg ${predNeg} \
    --lossMode ${lossMode} \
    --isCategorySpecific ${isCategorySpecific} \
    --save ${save} \
    --prototypeMode ${prototypeMode} \
    --ignoreMode ${ignoreMode} \
    --ignoreMargin ${ignoreMargin} \
    --selectRatio ${selectRatio} \
    --clsLossWeight ${clsLossWeight} \
    --instanceSampleMode ${instanceSampleMode} \
    --exampleSampleMode ${exampleSampleMode} \
    --mixMode ${mixMode} \
    --noise ${noise} \
    --noiseMode ${noiseMode} \
    --dropEpoch ${dropEpoch} \
    --isDropCategorySpecific ${isDropCategorySpecific} \
    --deltaRel ${deltaRel} \
    --usePLEpoch ${usePLEpoch} \