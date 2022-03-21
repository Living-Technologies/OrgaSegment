#Init logger
import logging
import logging.config
logging.config.fileConfig('conf/logging.conf')
logger = logging.getLogger(__file__)

#Import Mask RCNN packages
import mrcnn.model as modellib
from mrcnn import utils

#Import OrgaSegment functions
from lib import OrganoidDataset, mask_projection, average_precision
import importlib

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

#Get Job ID
job_id=sys.argv[1]

#Get config
config_path=sys.argv[2]
spec = importlib.util.spec_from_file_location('EvalConfig', config_path)
modulevar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulevar)
config = modulevar.EvalConfig()

#Model
model_path=sys.argv[3]

#Set log_dir
log_dir = None

def main():
    #Get config, display and save config
    # config = EvalConfig()
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
        model_name = model_path
        model.load_weights(model_path, by_name=True)
        logger.info(f'Model loaded: {model_name}')
    else:
        model_name = model.find_last()
        model.load_weights(model_name, by_name=True)
        logger.info(f'Model loaded: {model_name}')
    
    #Update log_dir
    global log_dir
    log_dir = model.log_dir

    #Create empty data frame for results
    evaluation =  pd.DataFrame({'image': pd.Series([], dtype='str'),
                                'class': pd.Series([], dtype=np.int8),
                                'ap': pd.Series([], dtype='object'),
                                'tp': pd.Series([], dtype='object'),
                                'fp': pd.Series([], dtype='object'),
                                'fn': pd.Series([], dtype='object')})

    # Compute Average Precisions based on Cellpose paper
    for i in data_eval.image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(data_eval, config,
                                       i, use_mini_mask=False)
        
        # Run object detection
        results = model.detect([image], verbose=1)
        r = results[0]
        
        for class_id in list(set(gt_class_id)):
            #Get gt per class id
            gt_indices = [i for i, u in enumerate(gt_class_id) if u == class_id] #get gt indices where label is equal to i
            gt = [gt_mask[:,:,i] for i in gt_indices] #get gt masks where label is equal to i
            gt = mask_projection(np.stack(gt, axis=-1))

            #Get prediction per class id
            p_indices = [i for i, u in enumerate(r['class_ids']) if u == class_id] #get prediction indices where label is equal to i
            scores = [r['scores'][i] for i in p_indices] #get scores where label is equal to i
            p_masks = [r['masks'][:,:,i] for i in p_indices] #get prediction masks where label is equal to i

            #Remove masks with a low score
            s_indices = [i for i, u in enumerate(scores) if u >= config.CONFIDENCE_SCORE_THRESHOLD] #get prediction indices where score is higher than thresholds
            p = [p_masks[i] for i in s_indices] #get prediction masks where score is higher than threshold
            p = mask_projection(np.stack(p, axis=-1))

            # Compute AP
            ap, tp, fp, fn = average_precision(gt, p, config.AP_THRESHOLDS)
        
            #Combine information
            info = {'image': data_eval.info(i)['path'],
                    'class': class_id,
                    'ap': ap,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn}
            evaluation = evaluation.append(info, ignore_index=True)

    #Save results
    evaluation.to_csv(model_name.replace('.h5', '_evaluation.csv'), index=False)
        
if __name__ == "__main__":
    logger.info('Start evaluation...')
    main()
    logger.info('Evaluation completed!')
    ##Copy logging to model log dir
    shutil.copy(f'log/JobName.{job_id}.out', f'{log_dir}/JobName.{job_id}.out')
    shutil.copy(f'log/JobName.{job_id}.err', f'{log_dir}/JobName.{job_id}.err')