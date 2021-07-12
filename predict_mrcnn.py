#Init logger
import logging
import logging.config
logging.config.fileConfig('conf/logging.conf')
logger = logging.getLogger(__file__)

#Import Mask RCNN packages
import mrcnn.model as modellib

#Import OrgaSegment functions
from lib import get_image_names
from conf import PredictConfig

#Import other packages
import tensorflow as tf
import sys
import shutil
from skimage.io import imsave
import pandas as pd
import numpy as np
import random
import re
import os
from keras.preprocessing.image import load_img

#Set Tensorflow logging
logger.info(f'Tensorflow version: {tf.__version__}')
tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#Check Tensorflow GPU
if tf.test.is_gpu_available():
    logger.info(f'GPU devices: {tf.config.experimental_list_devices()}')
else:
    logger.error(f'No GPUs available')
    exit(1)

#Get Job ID
job_id=sys.argv[1]

#Data folder
data_dir=sys.argv[1]
if os.path.isdir(data_dir) == False:
            logger.error(f'Incorrect data path specified: {data_dir}')
            exit(1)
else:
    data_dir=os.path.join(data_dir, '')
    logger.info(f'Data dir: {data_dir}')

def main():
    #Get config, display and save config
    config = InferenceConfig()
    logger.info(config.display())

    #Get data
    logger.info('Get image names')
    images = get_image_names(data_dir, '_orgaseg_masks')

    #Load model
    logger.info('Loading model')
    model = modellib.MaskRCNN(mode='inference', 
                              config=config,
                              model_dir=config.MODEL_DIR)
    model.load_weights(config.MODEL, by_name=True)

    #Create empty data frame for results
    results =  pd.DataFrame({'image': pd.Series([], dtype='str'),
                            'mask': pd.Series([], dtype='str'),
                            'name': pd.Series([], dtype='str'),
                            'id': pd.Series([], dtype=np.int16),
                            'y1':  pd.Series([], dtype=np.int16),
                            'x1': pd.Series([], dtype=np.int16), 
                            'y2': pd.Series([], dtype=np.int16), 
                            'x2': pd.Series([], dtype=np.int16),
                            'class': pd.Series([], dtype=np.int16),
                            'score':  pd.Series([], dtype=np.float32),
                            'size': pd.Series([], dtype=np.int16)})

    #Run on images
    logger.info('Start predictions')
    for i in images:
        logger.info(f'Processing {i}')
        image_name = re.search(f'^{data_dir}(.*)\..*$', i).group(1)
        mask_name = f'{image_name}_orgaseg_masks.png'
        mask_path = data_dir + mask_name

        #Load image
        img = np.asarray(load_img(i, color_mode=config.COLOR_MODE))

        #Predict organoids
        pred = model.detect([img], verbose=1)
        p = pred[0]

        #Create length of predictions
        length = len(p['rois'])
        values = list(range(1, length + 1)) #Create a list of values from 1 to number of masks per label

        #Process predictions
        for count, l in enumerate(range(length)):
            #Get mask information
            msk = p['masks'][:,:,l].astype(np.uint8)
            size = np.sum(msk)
            # num = values.pop(random.randrange(len(values))) #Get a random number for mask
            num = l + 1
            msk = np.where(msk != 0, num, msk)
            if count == 0:
                mask = msk
            else:
                mask = np.maximum(mask, msk) #Combine previous mask with new mask

            #Set all information
            info = {'image': i,
                    'mask': mask_path,
                    'name': image_name,
                    'id': num,
                    'y1': p['rois'][l,0],
                    'x1': p['rois'][l,1],
                    'y2': p['rois'][l,2],
                    'x2': p['rois'][l,3],
                    'class': p['class_ids'][l],
                    'score': p['scores'][l],
                    'size': size}
            results = results.append(info, ignore_index=True)
        
        #Save mask
        imsave(mask_path, mask)

    #Save results
    results.to_csv(f'{data_dir}orgaseg_results.csv', index=False)
        
if __name__ == "__main__":
    logger.info('Start prediction job...')
    main()
    logger.info('Predictions completed!')
    ##Copy logging to data dir
    shutil.copy(f'log/JobName.{job_id}.out', f'{data_dir}/JobName.{job_id}.out')
    shutil.copy(f'log/JobName.{job_id}.err', f'{data_dir}/JobName.{job_id}.err')