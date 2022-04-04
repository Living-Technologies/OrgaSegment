#Logger: https://docs.python.org/3/howto/logging.html
#Init logger
import logging
import logging.config
logging.config.fileConfig('conf/logging.conf')
logger = logging.getLogger(__file__)

import numpy as np
import pandas as pd

import skimage.io
from cellpose import models
from lib import average_precision
import glob, os

#Import Neptune tracking
from dotenv import load_dotenv
import neptune.new as neptune

parameters = {'NAME': 'CELLPOSE-ORGANOIDS',
              'MODEL': 'cellpose_residual_on_style_on_concatenation_off_train_2022_03_28_10_21_44.281679',
              'model_path': '/hpc/umc_beekman/cellpose/datasets/20211206/train/models/cellpose_residual_on_style_on_concatenation_off_train_2022_03_28_10_21_44.281679',
              'class_id': 1,
              'class_name': 'organoid',
              'channels': [[0,0]],
              'input_dir': '/hpc/umc_beekman/cellpose/datasets/20211206/eval',
              'diameter': None,
              'ap_thresholds': np.arange(0.5, 1.05, 0.05)}
logger.info(parameters)

#Create neptune logger
load_dotenv()
run = neptune.init(project=os.getenv('NEPTUNE_PROJECT'),
                   api_token=os.getenv('NEPTUNE_APIKEY'),
                   name = parameters['MODEL'])      
run['parameters'] = parameters
run['sys/tags'].add(['evaluate'])
run['dataset/eval'].track_files(parameters['input_dir'])

#Load model
logger.info('Load model')
model = models.CellposeModel(pretrained_model=parameters['model_path'])

#Load images and masks
image_names = []
mask_names = []

image_names.extend(glob.glob(parameters['input_dir'] + '/*%s.jpg'%'_img'))
mask_names.extend(glob.glob(parameters['input_dir'] + '/*%s.png'%'_masks_organoid'))

logger.info('Load images')
imgs = [skimage.io.imread(f) for f in image_names]
logger.info(f'Number of images: {len(imgs)}')

logger.info('Load GT masks')
gt_msks = [skimage.io.imread(f) for f in mask_names]
logger.info(f'Number of GT masks: {len(gt_msks)}')

# define CHANNELS to run segementation on
# grayscale=0, R=1, G=2, B=3
# channels = [cytoplasm, nucleus]
# if NUCLEUS channel does not exist, set the second channel to 0
# IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
# channels = [0,0] # IF YOU HAVE GRAYSCALE
# channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
# channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

# if diameter is set to None, the size of the cells is estimated on a per image basis
# you can set the average cell `diameter` in pixels yourself (recommended)
# diameter can be a list or a single number for all images

#Create empty data frame for results
evaluation =  pd.DataFrame({'image': pd.Series([], dtype='str'),
                            'class_id': pd.Series([], dtype=np.double),
                            'class_name': pd.Series([], dtype='str'),
                            'threshold': pd.Series([], dtype=np.double),
                            'ap': pd.Series([], dtype=np.double),
                            'tp': pd.Series([], dtype=np.double),
                            'fp': pd.Series([], dtype=np.double),
                            'fn': pd.Series([], dtype=np.double)})

class_name = parameters['class_name']
class_id = parameters['class_id'] 

for index, item in enumerate(imgs):
    
    logger.info(f'Predict image: {index}')
    masks, flows, styles = model.eval([item], diameter=parameters['diameter'], channels=parameters['channels'])
    ap, tp, fp, fn = average_precision(gt_msks[index], masks[0], parameters['ap_thresholds'])
    logger.info(f'Thresholds: parameters["ap_thresholds"]')
    logger.info(f'Image: {index}; APs: {ap}; TPs: {tp}; FPs: {fp}; FNs: {fn}')

    #Combine information
    for t in range(parameters['ap_thresholds'].size):
        info = {'image': image_names[index],
                'threshold': round(parameters['ap_thresholds'][t], 2),
                'ap': ap[t],
                'tp': tp[t],
                'fp': fp[t],
                'fn': fn[t]}
        logger.info(f'INFO: {info}')
        evaluation = evaluation.append(info, ignore_index=True)

        run[f'eval/images/ID_{index}'] = image_names[index]
        #AP
        run[f'eval/class_name={class_name}/ID_{index}/ap'].log(value=ap[t], step=round(parameters['ap_thresholds'][t], 2))
        run[f'eval/class_name={class_name}/ID_{index}/ap@{round(parameters["ap_thresholds"][t], 2)}'] = ap[t]
        #TP
        run[f'eval/class_name={class_name}/ID_{index}/tp'].log(value=tp[t], step=round(parameters['ap_thresholds'][t], 2))
        run[f'eval/class_name={class_name}/ID_{index}/tp@{round(parameters["ap_thresholds"][t], 2)}'] = tp[t]
        #FP
        run[f'eval/class_name={class_name}/ID_{index}/fp'].log(value=fp[t], step=round(parameters['ap_thresholds'][t], 2))
        run[f'eval/class_name={class_name}/ID_{index}/fp@{round(parameters["ap_thresholds"][t], 2)}'] = fp[t]
        #FN
        run[f'eval/class_name={class_name}/ID_{index}/fn'].log(value=fn[t], step=round(parameters['ap_thresholds'][t], 2))
        run[f'eval/class_name={class_name}/ID_{index}/fn@{round(parameters["ap_thresholds"][t], 2)}'] = fn[t]


summary = evaluation.groupby(['class_name', 'threshold'], as_index=False)['ap'].mean()
logger.info(f'Summary {summary}')
for i in range(len(summary)):
    run[f'eval/class_name={summary["class_name"][i]}/mAP'].log(value=summary['ap'][i], step=summary['threshold'][i])
    run[f'eval/class_name={summary["class_name"][i]}/mAP@{summary["threshold"][i]}'] = summary['ap'][i]

#Save results
evaluation.to_csv(parameters['model_path'].replace('.281679', '_evaluation.csv'), index=False)
    
run.stop()