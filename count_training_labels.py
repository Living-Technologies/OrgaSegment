import glob
from logging.config import valid_ident
import numpy as np
from keras.preprocessing.image import load_img

dirName = '/mnt/l/researchklz/KLZ-LAB_21-000_ORGASEGMENT/E_ResearchData/1_Datasets/basic_20211206/'
train = [f for f in glob.glob(dirName + "train/*masks_organoid.png", recursive=True)]
val = [f for f in glob.glob(dirName + "val/*masks_organoid.png", recursive=True)]
eval = [f for f in glob.glob(dirName + "eval/*masks_organoid.png", recursive=True)]

print('TRAIN')
trainLabels = []
for f in train:
    mask = np.asarray(load_img(f))
    
    trainLabels.append(np.amax(mask))
print(len(trainLabels))
print(sum(trainLabels))

print('VAL')
valLabels = []
for f in val:
    mask = np.asarray(load_img(f))
    
    valLabels.append(np.amax(mask))
print(len(valLabels))
print(sum(valLabels))

print('EVAL')
evalLabels = []
for f in eval:
    mask = np.asarray(load_img(f))
    
    evalLabels.append(np.amax(mask))
print(len(evalLabels))
print(sum(evalLabels))

print('TOTAL')
total = trainLabels + valLabels + evalLabels
print(len(total))
print(sum(total))
