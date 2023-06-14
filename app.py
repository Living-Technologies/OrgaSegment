import streamlit as st
import tkinter as tk
from tkinter import filedialog
import os
import configparser

#Import Mask RCNN packages
import mrcnn.model as modellib
# import tensorflow as tf

#Import OrgaSegment functions
from lib import get_image_names, mask_projection, display_preview
import importlib

#Import other packages
from skimage.io import imsave
import pandas as pd
import numpy as np
import trackpy as tp
import re
import os
from pathlib import Path
from keras.preprocessing.image import load_img

#Get app config
config = configparser.ConfigParser()
config.sections()
config.read('./conf/app.conf')
config.sections()

# Check if 'input_dir' already exists in session_state
# If not, then initialize it
if 'model' not in st.session_state:
    st.session_state['model'] = 'None'
if 'model_config' not in st.session_state:
    st.session_state['model_config'] = 'None'
if 'model_path' not in st.session_state:
    st.session_state['model_path'] = 'None'
if 'input_dir' not in st.session_state:
    st.session_state['input_dir'] = 'None'


# Set up tkinter
root = tk.Tk()
root.withdraw()
# Make folder picker dialog appear on top of other windows
root.wm_attributes('-topmost', 1)

#####Create app
st.title('OrgaSegment')
nameLocation = st.empty()
imageLocation = st.empty()
st.sidebar.header('Settings')
st.sidebar.subheader('Select model')
st.session_state['model'] = st.sidebar.selectbox('Please select model for segmentation', config.sections())
st.session_state['model_config'] = config[st.session_state['model']]['config']
st.session_state['model_path'] = config[st.session_state['model']]['model']
st.sidebar.text_input('Selected model:', st.session_state['model'])
#Get model config
spec = importlib.util.spec_from_file_location('PredictConfig', st.session_state['model_config'])
modulevar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulevar)
st.session_state['model_config'] = modulevar.PredictConfig()

st.sidebar.subheader('Select options')
st.session_state['predict'] = st.sidebar.checkbox('Interference')
st.session_state['track'] = st.sidebar.checkbox('Track')

st.sidebar.subheader('Select data folder')
if st.sidebar.button('Folder Picker'):
    st.session_state['input_dir'] = filedialog.askdirectory(master=root)
st.sidebar.text_input('Selected folder:', st.session_state['input_dir'])

if st.session_state['track']:
    st.sidebar.subheader('Select track options')
    st.session_state['regex'] = st.sidebar.text_input('File name regex, correct if needed',  '.*(?P<WELL>[A-Z]{1}[0-9]{1,2}).*t(?P<T>[0-9]{1,2}).*')
    st.session_state['search_range'] = st.sidebar.text_input('Tracking search range in pixels, correct if needed',  '50')
    st.session_state['memory'] = st.sidebar.text_input('Memory: the maximum number of frames duing which an organoid can vanisch, then reappear within the search range, and be considered the same organoid. Correct if needed',  '0')

st.sidebar.subheader('Run application')
if st.sidebar.button('Run'):
    progress_bar = st.sidebar.progress(0)
    #Get data
    images = get_image_names(st.session_state['input_dir'], '_masks')

    #Create folders
    input_dir=os.path.join(st.session_state['input_dir'], '')
    output_dir=os.path.join(input_dir, st.session_state['model'], '')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    preview_dir=os.path.join(input_dir, st.session_state['model'], 'preview', '')
    Path(preview_dir).mkdir(parents=True, exist_ok=True)

    if st.session_state['predict']:
        #Load model
        model = modellib.MaskRCNN(mode='inference', 
                                config=st.session_state['model_config'],
                                model_dir='./models')
        model.load_weights(st.session_state['model_path'], by_name=True)

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
        for image_count, i in enumerate(images):
            try:
                if os.name == 'nt':
                    image_name = re.search(f'^{input_dir}\(.*)\..*$', i).group(1)
                else: 
                    image_name = re.search(f'^{input_dir}(.*)\..*$', i).group(1)

                #Load image
                img = np.asarray(load_img(i, color_mode=st.session_state['model_config'].COLOR_MODE))
                if np.amax(img) >= 255:
                    img = ((img - np.amax(img)) * 255).astype('uint8')

                if st.session_state['model_config'].COLOR_MODE == 'grayscale':
                    img = img[..., np.newaxis]

                #Predict organoids
                pred = model.detect([img], verbose=1)
                p = pred[0]
                
                #Combine image and mask and create preview
                preview_name = f'{image_name}_preview.png'
                preview_path = preview_dir + preview_name
                preview = display_preview(np.asarray(load_img(i, color_mode='rgb')), 
                                          p['rois'],
                                          p['masks'], 
                                          p['class_ids'],  
                                          st.session_state['model_config'].CLASSES, 
                                          p['scores'],
                                          figsize=(40, 40))
                preview.savefig(preview_path, format='png', dpi=350, bbox_inches='tight', pad_inches=0)
                
                nameLocation.subheader(f'Image: {image_name}')
                imageLocation.image(load_img(preview_path, color_mode='rgb'), use_column_width=True)

                #Process results per class
                for c in np.unique(p['class_ids']):
                    #Create names
                    mask_name = f'{image_name}_masks_class-{c}.png'
                    mask_path = output_dir + mask_name

                    #Get mask
                    unique_class_ids = (p['class_ids'] == c).nonzero()[0]
                    mask = mask_projection(p['masks'][:,:,unique_class_ids])

                    #Save mask
                    imsave(mask_path, mask)

                    #Process predictions
                    for count, l in enumerate(unique_class_ids):
                        #Get mask information
                        msk = p['masks'][:,:,l].astype(np.uint8)
                        size = np.sum(msk)

                        #Set all information
                        info = {'image': i,
                                'mask': mask_path,
                                'name': image_name,
                                'id': count,
                                'y1': p['rois'][l,0],
                                'x1': p['rois'][l,1],
                                'y2': p['rois'][l,2],
                                'x2': p['rois'][l,3],
                                'class': p['class_ids'][l],
                                'score': p['scores'][l],
                                'size': size}
                        results = results.append(info, ignore_index=True)
                progress_bar.progress((image_count+1)/len(images))
            except:
               pass

        #Save results
        results.to_csv(f'{output_dir}results.csv', index=False)

    #Track
    if st.session_state['track']:
        #Get data
        results = pd.read_csv(f'{output_dir}results.csv')

        #Enrich data
        results['well'] = results['name'].apply(lambda x: re.search(st.session_state['regex'], x).group('WELL'))
        results['t'] = results['name'].apply(lambda x: re.search(st.session_state['regex'], x).group('T'))
        
        ## Calculate centers and track organoids over time
        results['x'] = (results['x2'] + results['x1']) / 2
        results['y'] = (results['y2'] + results['y1']) / 2
        results = results.groupby('well').apply(tp.link, search_range=int(st.session_state['search_range']), memory=int(st.session_state['memory']), t_column='t').reset_index(drop=True)
        
        #Save results
        results.to_csv(f'{output_dir}tracked.csv', index=False)
    
    st.sidebar.subheader('Done!')

else:
    st.sidebar.text("Click Run to process all images.")