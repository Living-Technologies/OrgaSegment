#Init logger
import logging
import logging.config
logging.config.fileConfig('conf/logging.conf')
logger = logging.getLogger(__file__)

#Import Mask RCNN packages
import mrcnn.model as modellib

#Import OrgaSegment functions
from lib import get_image_names, OrganoidDataset
from conf import InferenceConfig

#Import other packages
import tensorflow as tf
import sys
import shutil
from skimage.io import imsave
import pandas as pd
import numpy as np
import random
from keras.preprocessing.image import load_img

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
    logger.info('Get image names')
    images = get_image_names(config.INFERENCE_DIR, '_orgaseg_masks', config.IMAGE_FILTER)

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
        image_name = re.search(f'^{config.INFERENCE_DIR}(.*)\..*$', i).group(1)
        mask_name = f'{image_name}_orgaseg_masks.png'
        mask_path = config.INFERENCE_DIR + mask_name

        #Load image
        img = load_img(i, color_mode=config.COLOR_MODE)

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
            num = values.pop(random.randrange(len(values))) #Get a random number for mask
            msk = np.where(msk != 0, num, msk)
            if count == 0:
                mask = msk
            else:
                mask = np.maximum(mask, msk) #Combine previous mask with new mask

            #Set all information
            info = {'image': i,
                    'mask': mask_path,
                    'name': image_name,
                    'mask_id': num,
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
    results.to_csv(f'{config.INFERENCE_DIR}orgaseg_results.csv', index=False)
        
if __name__ == "__main__":
    logger.info('Start inference...')
    main()
    logger.info('Inference completed!')
    ##Copy logging to model log dir
    shutil.copy(f'log/JobName.{job_id}.out', f'{log_dir}/JobName.{job_id}.out')
    shutil.copy(f'log/JobName.{job_id}.err', f'{log_dir}/JobName.{job_id}.err')