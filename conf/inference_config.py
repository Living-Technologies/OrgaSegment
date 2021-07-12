import TrainConfig
import os

##Config
class InferenceConfig(TrainConfig):
    '''
    Configuration for Inference on the Organoid dataset.
    Derives from the base Config class and overrides values specific
    to the organoid dataset.
    '''
    # Number of GPU, images per GPU and batchsize
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Input image resizing
    IMAGE_RESIZE_MODE = 'none'

    # OrgaSegment specific config
    MODEL_DIR = os.path.join('/hpc/umc_beekman/orgasegment/models/', '')
    MODEL = '/hpc/umc_beekman/orgasegment/models/organoids20210708T1343/mask_rcnn_organoids_0500.h5'
    INFERENCE_DIR= os.path.join('/hpc/umc_beekman/labelbox_organoid_labels/dataset_organoids/20210701/test/', '')
    IMAGE_FILTER = '_img'
    # INFERENCE_DIR= os.path.join('/hpc/umc_beekman/DIS/20210526/DIS', '')
    # IMAGE_FILTER = ''
    CLASSES = ['organoid']
    COLOR_MODE = 'rgb'