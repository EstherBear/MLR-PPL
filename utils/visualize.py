import numpy as np
import matplotlib.pyplot as plt
import os

post = '0615_Noisy0.3pos_COCO_Semantic_BCELoss_EarlyStop_Epoch1'
statisticsDir = 'exp/statistics'
visDir = '{}/{}'.format('exp/vis', post)
PN = '_PN.npy'
PP = '_PP.npy'
PP_PN = '_PP-PN.npy'
PTN = '_PTN.npy'
PFN = '_PFN.npy'
negDropRate = '_negDropRate.npy'
PPList = np.zeros((20, 80))
PTNList = np.zeros((20, 80))
PFNList = np.zeros((20, 80))
for epoch in range(20):
    fileName = '{}/{}/{}{}'.format(statisticsDir, post, str(epoch), PP)
    PPList[epoch] = np.load(fileName)
    fileName = '{}/{}/{}{}'.format(statisticsDir, post, str(epoch), PTN)
    PTNList[epoch] = np.load(fileName)
    fileName = '{}/{}/{}{}'.format(statisticsDir, post, str(epoch), PFN)
    PFNList[epoch] = np.load(fileName)
if not os.path.exists(visDir):
    os.mkdir(visDir)
for classIdx in range(80):
    plt.plot(np.arange(20).astype(dtype=np.str), PPList[:, classIdx], label='Positive Prediction')
    plt.plot(np.arange(20).astype(dtype=np.str), PTNList[:, classIdx], label='True Negative Prediction')
    plt.plot(np.arange(20).astype(dtype=np.str), PFNList[:, classIdx], label='False Negative Prediction')
    plt.title('Category {}'.format(classIdx))
    plt.xlabel('Epochs')
    plt.ylabel('Prediction')
    plt.legend()
    plt.savefig(fname='{}/{}_PP_PTN_PFN.png'.format(visDir, str(classIdx)))
    plt.clf()
# fileName = '{}/{}/{}{}'.format(statisticsDir, post, str(epoch), PP_PN)
# print(np.load(fileName))