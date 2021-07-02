from mrcnn.config import Config

##Config
class SegmentConfig(Config):
    '''
    Configuration for training on the Organoid dataset.
    Derives from the base Config class and overrides values specific
    to the organoid dataset.
    '''
    ## MASK RCNN specific Config
    # Give the configuration a recognizable name
    NAME = 'ORGANOIDS'

    # Backbone
    BACKBONE = 'resnet50'
    
    # Number of GPU, images per GPU and batchsize
    GPU_COUNT = 2
    IMAGES_PER_GPU = 2
    BATCHSIZE = GPU_COUNT * IMAGES_PER_GPU

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 2 classes (organoid and unhealthy structure)

    # Set image dimensions for rescaling
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    # RPN ANCHOR scales
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    
    # TRAIN ROIS PER IMAGE
    TRAIN_ROIS_PER_IMAGE = 10

    # Steps per epoch and validation steps
    STEPS_PER_EPOCH = 50
    VALIDATION_STEPS = 5

    # OrgaSegment specific config
    TRAIN_DIR= '/hpc/umc_beekman/labelbox_organoid_labels/dataset_organoids/20210701/train'
    VAL_DIR= '/hpc/umc_beekman/labelbox_organoid_labels/dataset_organoids/20210701/val'
    MODEL_DIR='/hpc/umc_beekman/orgasegment/models/'
    IMAGE_FILTER = '_img'
    MASK_FILTER = '_masks_'
    CLASSES = ['organoid', 'unhealthy_structure']
    COLOR_MODE = 'rgb'