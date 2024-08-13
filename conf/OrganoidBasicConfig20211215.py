from mrcnn.config import Config
import numpy as np
import multiprocessing
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

    #Epochs
    EPOCHS_HEADS = 100
    EPOCHS_ALL_LAYERS = 500

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 class (organoid)

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = 'crop'
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 0

    #Grayscale channel count
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
    MINI_MASK_SHAPE = (64,64)

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200
    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400

    # Steps per epoch and validation steps
    STEPS_PER_EPOCH = 50
    VALIDATION_STEPS = 5

    #Multiprocessing
    #WORKERS = 1
    WORKERS = multiprocessing.cpu_count()
    MULTIPROCESSING = True

    # OrgaSegment specific config
    TRAIN_DIR = os.path.join('./data/trainTest/train', '')
    VAL_DIR = os.path.join('./data/trainTest/val', '')
    MODEL_DIR = os.path.join('./models/', '')
    PRETRAINED_WEIGHTS = './models/coco/mask_rcnn_coco.h5'
    EXCLUDE_LAYERS = ['conv1', 'mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask']
    IMAGE_FILTER = '_img'
    MASK_FILTER = '_masks_'
    CLASSES = ['organoid']
    COLOR_MODE = 'grayscale'
    EVAL_IOU = 0.75

##EvalConfig
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
    EVAL_DATASET = '20211206'
    EVAL_DIR = os.path.join(f'./data/{EVAL_DATASET}/eval', '')

    #Thresholds
    CONFIDENCE_SCORE_THRESHOLD = 0.0
    AP_THRESHOLDS = np.arange(0.5, 1.05, 0.05)

class PredictConfig(TrainConfig):
    '''
    Configuration for Prediction on a datsaset.
    Derives from the base Config class and overrides values specific
    to the organoid dataset.
    '''
    # Number of GPU, images per GPU and batchsize
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Input image resizing
    IMAGE_RESIZE_MODE = 'pad64'

    # OrgaSegment specific config
    MODEL_DIR = './models/'
    MODEL_NAME = 'OrganoidBasic20211215'
    MODEL = './models/' + MODEL_NAME + '.h5'
    COLOR_MODE = 'grayscale'

class TrackConfig():
    '''
    Configuration for Tracking organoids over time using Trackpy.
    '''
    # Regex to extract WELL (multiplate well position) and T (time) information from image name, correct if needed.
    REGEX = '.*(?P<WELL>[A-Z]{1}[0-9]{1,2}).*t(?P<T>[0-9]{1,2}).*'

    # Tracking search range in pixels, correct if needed.
    SEARCH_RANGE = 50

    # Memory: the maximum number of frames duing which an organoid can vanisch, then reappear within the search range, and be considered the same organoid. Correct if needed.
    MEMORY = 2