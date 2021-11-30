from mrcnn.config import Config
import numpy as np
import os

##Config
class TrainConfig(Config):
    '''
    Configuration for training on the Organoid dataset.
    Derives from the base Config class and overrides values specific
    to the organoid dataset.
    '''
    ## MASK RCNN specific Config
    # Give the configuration a recognizable name
    NAME = 'ORGANOIDS'

    # Backbone
    BACKBONE = 'resnet101'
    
    # Number of GPU, images per GPU and batchsize
    GPU_COUNT = 4
    IMAGES_PER_GPU = 1
    BATCHSIZE = GPU_COUNT * IMAGES_PER_GPU

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 #2  # background + 2 classes (organoid and unhealthy structure)

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = 'crop'
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    #Graysclae channel count
    IMAGE_CHANNEL_COUNT = 1

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD=0.99

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128

    # Image mean (RGB)
    # MEAN_PIXEL = np.array([126,126,126])
    
    #Image mean (Grayscale)
    MEAN_PIXEL = np.array([0])

    # RPN ANCHOR scales
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    
    # TRAIN ROIS PER IMAGE
    TRAIN_ROIS_PER_IMAGE = 128

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (100,100)

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200
    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400

    # Steps per epoch and validation steps
    STEPS_PER_EPOCH = 50
    VALIDATION_STEPS = 5

    # OrgaSegment specific config
    TRAIN_DIR = os.path.join('/hpc/umc_beekman/labelbox_organoid_labels/dataset_organoids/20211130/train', '')
    VAL_DIR = os.path.join('/hpc/umc_beekman/labelbox_organoid_labels/dataset_organoids/20211130/val', '')
    MODEL_DIR = os.path.join('/hpc/umc_beekman/orgasegment/models/', '')
    PRETRAINED_WEIGHTS = '/hpc/umc_beekman/orgasegment/models/coco/mask_rcnn_coco.h5'
    EXCLUDE_LAYERS = ['conv1', 'mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask']
    IMAGE_FILTER = '_img'
    MASK_FILTER = '_masks_'
    CLASSES = ['organoid'] #, 'unhealthy_structure']
    COLOR_MODE = 'grayscale'
    # COLOR_MODE = 'rgb'
    EVAL_IOU = 0.75
