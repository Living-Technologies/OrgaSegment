#Init logger
import logging
import logging.config
logging.config.fileConfig('conf/logging.conf')
logger = logging.getLogger(__file__)

#Import Mask RCNN packages
import mrcnn.model as modellib

#Import OrgaSegment functions
from lib import OrganoidDataset
from conf import InferenceConfig

#Import other packages
import tensorflow as tf
import sys
import shutil
from skimage.io import imsave
import pandas as pd
import numpy as np
import random

#Set Tensorflow logging
logger.info(f'Tensorflow version: {tf.__version__}')
tf.logging.set_verbosity(tf.logging.ERROR)

#Check Tensorflow GPU
if tf.test.is_gpu_available():
    logger.info(f'GPU devices: {tf.config.experimental_list_devices()}')
else:
    logger.error(f'No GPUs available')
    exit(1)

#Get Job ID
job_id=sys.argv[1]
#Set log_dir
log_dir = None

def main():
    #Get config, display and save config
    config = InferenceConfig()
    logger.info(config.display())

    #Update log_dir
    global log_dir
    log_dir = config.INFERENCE_DIR

    #Get data
    logger.info('Preparing data')
    data = OrganoidDataset()
    data.load_data(config.INFERENCE_DIR,
                   config.CLASSES,
                   config.IMAGE_FILTER,
                   config.MASK_FILTER,
                   config.COLOR_MODE)
    data.prepare()

    #Load model
    logger.info('Loading model')
    
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode='inference', 
                              config=config,
                              model_dir=config.MODEL_DIR)
    model.load_weights(config.MODEL, by_name=True)


    results =  pd.DataFrame({'image': pd.Series([], dtype='str'),
                          'mask': pd.Series([], dtype='str'),
                          'name': pd.Series([], dtype='str'),
                          'y1':  pd.Series([], dtype=np.int16),
                          'x1': pd.Series([], dtype=np.int16), 
                          'y2': pd.Series([], dtype=np.int16), 
                          'x2': pd.Series([], dtype=np.int16),
                          'class': pd.Series([], dtype=np.int16)})

    #Run on images
    logger.info('Start predictions')
    for i in data.image_ids:
        logger.info(f'Processing {data.info(i)["id"]}')
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(data,
                               config, 
                               i, 
                               use_mini_mask=False)
        
        #Create mask
        len_masks = gt_mask.shape[-1]
        values = list(range(1, len_masks + 1)) #Create a list of values from 1 to number of masks per label
        mask_name = f'{config.INFERENCE_DIR}{data.info(i)["id"]}_orgaseg_masks.png'

        #Process masks
        for count, m in enumerate(range(len_masks)):
            #Get mask information
            msk = gt_mask[:,:,m].astype(np.uint8)
            size = np.sum(msk)
            num = values.pop(random.randrange(len(values))) #Get a random number for mask
            msk = np.where(msk != 0, num, msk)
            if count == 0:
                mask = msk
            else:
                mask = np.maximum(mask, msk) #Combine previous mask with new mask

            #Set all information
            info = {'image': data.info(i)['path'],
                    'mask': mask_name,
                    'name': data.info(i)['id'],
                    'mask_id': num,
                    'y1': gt_bbox[m,0],
                    'x1': gt_bbox[m,1],
                    'y2': gt_bbox[m,2],
                    'x2': gt_bbox[m,3],
                    'class': gt_class_id[m],
                    'size': size}
            results = results.append(info, ignore_index=True)
        
        #Save mask
        imsave(mask_name, mask)

    #Save results
    results.to_csv(f'{config.INFERENCE_DIR}orgaseg_results.csv', index=False)
        
if __name__ == "__main__":
    logger.info('Start inference...')
    main()
    logger.info('Inference completed!')
    ##Copy logging to model log dir
    shutil.copy(f'log/JobName.{job_id}.out', f'{log_dir}/JobName.{job_id}.out')
    shutil.copy(f'log/JobName.{job_id}.err', f'{log_dir}/JobName.{job_id}.err')