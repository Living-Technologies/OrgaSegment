from conf import TrainConfig
import os
import numpy as np

##Config
class EvalConfig(TrainConfig):
    '''
    Configuration for Prediction on a datsaset.
    Derives from the base Config class and overrides values specific
    to the organoid dataset.
    '''
    # Number of GPU, images per GPU and batchsize
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = 'none'

    #Eval DIR
    EVAL_DIR = os.path.join('/hpc/umc_beekman/labelbox_organoid_labels/dataset_organoids/20210720/eval', '')

    #Thresholds
    CONFIDENCE_SCORE_THRESHOLD = 0.9
    AP_THRESHOLDS = np.arange(0.5, 1.05, 0.05)
