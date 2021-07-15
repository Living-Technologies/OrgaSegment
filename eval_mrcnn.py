#Init logger
import logging
import logging.config
logging.config.fileConfig('conf/logging.conf')
logger = logging.getLogger(__file__)

#Import Mask RCNN packages
import mrcnn.model as modellib
from mrcnn import utils

#Import OrgaSwell functions
from lib import OrganoidDataset
from conf import TrainConfig

#Import other packages
import tensorflow as tf
import sys
import os
import shutil
import pandas as pd
import numpy as np

#Set Tensorflow logging
logger.info(f'Tensorflow version: {tf.__version__}')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#Check Tensorflow GPU
if tf.test.is_gpu_available():
    logger.info(f'GPU devices: {tf.config.experimental_list_devices()}')
else:
    logger.error(f'No GPUs available')
    exit(1)

#Set config
class EvalConfig(TrainConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = 'none'

#Get Job ID
job_id=sys.argv[1]

#Model
model_path=sys.argv[2]

#Set log_dir
log_dir = None

def main():
    #Get config, display and save config
    config = EvalConfig()
    logger.info(config.display())

    #Get data
    logger.info('Preparing data')
    data_eval = OrganoidDataset()
    data_eval.load_data(config.EVAL_DIR,
                        config.CLASSES,
                        config.IMAGE_FILTER,
                        config.MASK_FILTER,
                        config.COLOR_MODE)
    data_eval.prepare()

    #Load model
    logger.info('Loading model')
    model = modellib.MaskRCNN(mode='inference', 
                              config=config,
                              model_dir=config.MODEL_DIR)
    
    if os.path.isfile(model_path) and model_path.endswith('.h5'):
        model.load_weights(model_path, by_name=True)
        logger.info(f'Model loaded: {model_path}')
    else:
        last_model = model.find_last()
        model.load_weights(last_model, by_name=True)
        logger.info(f'Model loaded: {last_model}')
    
    #Update log_dir
    global log_dir
    log_dir = model.log_dir

    #Create empty data frame for results
    evaluation =  pd.DataFrame({'image': pd.Series([], dtype='str'),
                                'AP': pd.Series([], dtype=np.float32),
                                'precisions': pd.Series([], dtype='object'),
                                'recalls': pd.Series([], dtype='object'),
                                'overlaps': pd.Series([], dtype='object')})

    # Compute VOC-Style mAP @ IoU=0.75
    for i in data_eval.image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(data_eval, config,
                                       i, use_mini_mask=False)
        # molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
        
        # Run object detection
        results = model.detect([image], verbose=1)
        r = results[0]
        
        # Compute
        AP, precisions, recalls, overlaps =\
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                 r["rois"], r["class_ids"], r["scores"], r['masks'],
                                 config.EVAL_IOU)
    
        info = {'image': data_eval.info(i)['path'],
                'AP': AP,
                'precisions': precisions,
                'recalls': recalls,
                'overlaps': overlaps}

        evaluation = evaluation.append(info, ignore_index=True)

    #Log mean AP
    logger.info(f'Model: {model_path} || mAP @ IoU 0.75: {evaluation["AP"].mean()}')

    #Save results
    results.to_csv(model_path.replace('.h5', '_evaluation.csv'), index=False)
        
if __name__ == "__main__":
    logger.info('Start evaluation...')
    main()
    logger.info('Evaluation completed!')
    ##Copy logging to model log dir
    shutil.copy(f'log/JobName.{job_id}.out', f'{log_dir}/JobName.{job_id}.out')
    shutil.copy(f'log/JobName.{job_id}.err', f'{log_dir}/JobName.{job_id}.err')