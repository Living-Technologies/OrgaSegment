from conf import TrainConfig

##Config
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
    IMAGE_MAX_DIM = 2048 #To prevent OOM on GPU

    # OrgaSegment specific config
    MODEL_DIR = '/hpc/umc_beekman/orgasegment/models/'
    MODEL = '/hpc/umc_beekman/orgasegment/models/organoids20210721T1013/mask_rcnn_organoids_0500.h5'
    COLOR_MODE = 'grayscale'