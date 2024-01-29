#Init logger
import logging
import logging.config
logging.config.fileConfig('conf/logging.conf')
logger = logging.getLogger(__file__)

#Import Mask RCNN packages
import mrcnn.model as modellib

#Import OrgaSwell functions
from lib import OrganoidDataset, config_to_dict
import importlib

#Import other packages
import tensorflow as tf
import sys
import shutil
import os

# Instead of importlib etc.
import oseg_v1_conf



config = oseg_v1_conf.TrainConfig()
print(config.GPU_COUNT)
config.GPU_COUNT=1

def main():

    #Get data
    data_train = OrganoidDataset()
    data_train.load_data(config.TRAIN_DIR,
                         config.CLASSES,
                         config.IMAGE_FILTER,
                         config.MASK_FILTER,
                         config.COLOR_MODE)
    data_train.prepare()

    data_val = OrganoidDataset()
    data_val.load_data(config.VAL_DIR,
                       config.CLASSES,
                       config.IMAGE_FILTER,
                       config.MASK_FILTER,
                       config.COLOR_MODE)
    data_val.prepare()

    #Compile model
    model = modellib.MaskRCNN(mode='training', config=config, model_dir=config.MODEL_DIR)
    print(config.PRETRAINED_WEIGHTS)
    config.PRETRAINED_WEIGHTS="models/OrganoidBasic20211215.h5"
    config.EXCLUDE_LAYERS.append("anchors")
    print(config.EXCLUDE_LAYERS)
    for i, f in enumerate(model.keras_model.layers):
        print(i, f.name)

    model.load_weights(config.PRETRAINED_WEIGHTS,
                       by_name=True,
                       exclude=config.EXCLUDE_LAYERS)

    #Update log_dir
    global log_dir
    log_dir = model.log_dir
    name = os.path.basename(log_dir)

    ##Train model
    logger.info('Start training heads')
    model.train(data_train, data_val,
                learning_rate=config.LEARNING_RATE,
                epochs=config.EPOCHS_HEADS,
                layers='heads')

    logger.info('Start training all layers')
    model.train(data_train, data_val,
                learning_rate=config.LEARNING_RATE,
                epochs=config.EPOCHS_ALL_LAYERS,
                layers='all',
                workers=config.WORKERS,
                use_multiprocessing=config.MULTIPROCESSING,
                class_weight=config.CLASS_WEIGHTS)

if __name__ == "__main__":
    main()
