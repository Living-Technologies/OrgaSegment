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


    bbox =  pd.DataFrame({'image': pd.Series([], dtype='str'),
                          'mask': pd.Series([], dtype='str'),
                          'name': pd.Series([], dtype='str'),
                          'y1':  pd.Series([], dtype=np.int16),
                          'x1': pd.Series([], dtype=np.int16), 
                          'y2': pd.Series([], dtype=np.int16), 
                          'x2': pd.Series([], dtype=np.int16),
                          'class': pd.Series([], dtype=np.int16)})

    #Run on images
    for i in data.image_ids:
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(data,
                               config, 
                               i, 
                               use_mini_mask=False)
        
        #Create mask file
        len_masks = gt_mask.shape[-1]
        values = list(range(1, len_masks + 1)) #Create a list of values from 1 to number of masks per label
                
        for count, i in enumerate(range(len_masks)):
            msk = gt_mask[:,:,i].astype(int)
            num = values.pop(random.randrange(len(values))) #Get a random number for mask
            msk = np.where(msk != 0, num, msk)
            if count == 0:
                mask = msk
            else:
                mask = np.maximum(mask, msk) #Combine previous mask with new mask
        
        mask_name = f'{config.INFERENCE_DIR}{data.info(i)["id"]}_masks.png'
        #Save mask file
        imsave(mask_name)

        #Append bbox and class information
        for i in range(gt_bbox.shape[0]):
            bb = {'image': data.image_info(i)['path'],
                  'mask': mask_name,
                  'name': data.info(i)['id'],
                  'y1': gt_bbox[i,0],
                  'x1': gt_bbox[i,1],
                  'y2': gt_bbox[i,2],
                  'x2': gt_bbox[i,3],
                  'class': gt_class_id[i]}
            bbox = bbox.append(bb, ignore_index=True)
    
    #Save bbox information
    bbox.to_csv(f'{config.INFERENCE_DIR}bbox.csv', ignore_index=True)
        
if __name__ == "__main__":
    logger.info('Start training...')
    main()
    logger.info('Training completed!')
    ##Copy logging to model log dir
    shutil.copy(f'log/JobName.{job_id}.out', f'{log_dir}/JobName.{job_id}.out')
    shutil.copy(f'log/JobName.{job_id}.err', f'{log_dir}/JobName.{job_id}.err')