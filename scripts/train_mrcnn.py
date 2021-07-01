#Init logger
import logging
import logging.config
logging.config.fileConfig('conf/logging.conf')
logger = logging.getLogger(__file__)

#Temp
import os
print(f'Current working directory: {os.getcwd()}')

#Import Mask RCNN packages
import mrcnn.model as modellib
# from mrcnn import utils
# from mrcnn import visualize
# from mrcnn.model import log

print(f'Current working directory: {os.getcwd()}')
#Import OrgaSwell functions
from fun import OrganoidDataset
from conf import SegmentConfig

#Import other packages
import tensorflow as tf
import sys

#Set Tensorflow logging
logger.info(f'Tensorflow version: {tf.__version__}')
tf.logging.set_verbosity(tf.logging.ERROR)

#Check Tensorflow GPU
if tf.test.is_gpu_available():
    logger.error(f'No GPUs available')
    exit(1)
else:
    logger.info(f'Num GPUs Available: {len(tf.config.experimental_list_devices())}')

def main():
    #Get config, display and save config
    config = SegmentConfig()
    logger.info(config.display())

    #Get data
    logger.info('Preparing data')
    data_train = OrganoidDataset()
    data_train.load_data(config.TRAIN_DIR,
                         config.CLASSES,
                         config.IMAGE_FILTER,
                         config.MASK_FILTER,
                         (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]),
                         config.COLOR_MODE)
    data_train.prepare()

    data_val = OrganoidDataset()
    data_valload_data(config.VAL_DIR,
                      config.CLASSES,
                      config.IMAGE_FILTER,
                      config.MASK_FILTER,
                      (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]),
                      config.COLOR_MODE)
    data_val.prepare()

    #Compile model
    logger.info('Compiling model')
    model = modellib.MaskRCNN(mode='training', config=config, model_dir=config.MODEL_DIR)
    model.load_weights(model.get_imagenet_weights(), by_name=True)

    #Save config
    logger.info('Saving config')
    sys.stdout = open(f'{model.log_dir}/config.txt', "w")
    config.display()
    sys.stdout.close()
    
    ##Train model
    logger.info('Start training model')
    model.train(data_train, data_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=config.EPOCHS,
                layers=config.LAYERS)

if __name__ == "__main__":
    logger.info('Start training...')
    main()
    logger.info('Training completed!')