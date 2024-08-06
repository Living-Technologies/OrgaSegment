#!/usr/bin/env python


import sys

from tensorflow import keras
import mrcnn.model as model
import skimage.io as imageio

import numpy

# Instead of importlib etc.
import oseg_v1_conf

from matplotlib import pyplot

import time
import tensorflow.keras
import gc
import pathlib

def getModelPath():
    import importlib
    return importlib.resources.files("models").joinpath("OrganoidBasic20211215.h5")

def mergeLabels(masks):
    merged = numpy.zeros( (*masks.shape[:-1], 3), dtype="int8")
    for j in range(masks.shape[-1]):
        b = masks[:, :, j] > 0.9
        merged[:,:,0] |= (((j + 1)*172)%255)*b
        merged[:,:,1] |= ((j*339)%255)*b
        merged[:,:,2] |= ((j*569)%255)*b

    return merged

def loadImage( f ):
    img = imageio.imread( f )
    if len(img.shape) == 2:
        img = numpy.expand_dims(img, 2)
    else:
        img = numpy.max(img, axis=2, keepdims=True)
    print(img.shape)
    return img

def saveImage( f, arr ):
    arr = numpy.rollaxis(arr, 2)
    imageio.imsave(f, numpy.array(arr, dtype="int8"))

if __name__=="__main__":
    config = oseg_v1_conf.PredictConfig()
    print(config.NUM_CLASSES)
    mdl = model.MaskRCNN("inference", config, "model_path")
    #mdl.load_weights("oseg-v1.h5", by_name=True)
    #mdl.keras_model.load_weights("models/OrganoidBasic20211215.h5", by_name=True)
    mdl.keras_model.load_weights(getModelPath(), by_name=True)
    #mdl.keras_model.summary()
    print(mdl.keras_model.outputs)
    print(mdl.keras_model.input)
    ins = []
    outs = []
    cum = 0
    opth = pathlib.Path("oseg-predictions")
    if not opth.exists():
        opth.mkdir()

    for f in sys.argv[1:]:
        pth = pathlib.Path(f)

        img = loadImage(f)
        start = time.time()
        pred = mdl.detect([img] )
        elapsed = time.time() - start
        print("%s in %s"%(f, elapsed))
        cum += elapsed
        for p in pred:
            mgd = mergeLabels(p['masks'])
            ins.append(img)
            outs.append(mgd)
            #save here!
            nm = pth.name
            nm2 = "pred-%s_masks.png"%nm[:nm.rfind(".")]
            tp = pathlib.Path(opth, nm2)
            saveImage(tp, mgd)
        #tensorflow.keras.backend.clear_session()
        gc.collect()

    print("%s total time"%cum)
    for i, p in zip(ins, outs):
        pyplot.figure(0)
        pyplot.imshow(i)
        pyplot.figure(1)
        pyplot.imshow(p)
        pyplot.show()
