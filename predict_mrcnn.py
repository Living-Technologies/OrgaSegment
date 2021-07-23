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
import re
import os
from pathlib import Path
from keras.preprocessing.image import load_img

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

#Data folders
input_dir=sys.argv[2]
if os.path.isdir(input_dir) == False:
    logger.error(f'Incorrect input path specified: {input_dir}')
    exit(1)
else:
    input_dir=os.path.join(input_dir, '')
    logger.info(f'Input dir: {input_dir}')
    output_dir=os.path.join(input_dir, 'orgasegment', '')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f'Output dir: {output_dir}')

def main():
    #Get config, display and save config
    config = PredictConfig()
    logger.info(config.display())

    #Get data
    logger.info('Get image names')
    images = get_image_names(input_dir, '_masks')

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
        image_name = re.search(f'^{input_dir}(.*)\..*$', i).group(1)
        mask_name = f'{image_name}_masks.png'
        mask_path = output_dir + mask_name

        #Load image
        img = np.asarray(load_img(i, color_mode=config.COLOR_MODE))
        if config.COLOR_MODE == 'grayscale':
            img = img[..., np.newaxis]

        logger.info(f'IMG shape: {img.shape}')

        #Predict organoids
        pred = model.detect([img], verbose=1)
        p = pred[0]

        #Create length of predictions
        length = len(p['rois'])
        
        #Reset mask
        mask =  np.zeros((img.shape[0], img.shape[1]))

        logger.info(f'Init mask shape: {mask.shape}')

        #Process predictions
        for count, l in enumerate(range(length)):
            #Get mask information
            msk = p['masks'][:,:,l].astype(np.uint8)
            size = np.sum(msk)
            num = l + 1
            msk = np.where(msk != 0, num, msk)
            if count == 0:
                mask = msk
            else:
                mask = np.maximum(mask, msk) #Combine previous mask with new mask
            
            logger.info(f'Mask shape: {mask.shape} ')

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
    results.to_csv(f'{output_dir}results.csv', index=False)
        
if __name__ == "__main__":
    logger.info('Start prediction job...')
    main()
    logger.info('Predictions completed!')
    ##Copy logging to data dir
    shutil.copy(f'log/JobName.{job_id}.out', f'{output_dir}JobName.{job_id}.out')
    shutil.copy(f'log/JobName.{job_id}.err', f'{output_dir}JobName.{job_id}.err')