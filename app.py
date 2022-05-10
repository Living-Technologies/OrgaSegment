import streamlit as st
import tkinter as tk
from tkinter import filedialog
import os
import configparser

#Import Mask RCNN packages
import mrcnn.model as modellib

#Import OrgaSegment functions
from lib import get_image_names, mask_projection
import importlib

#Import other packages
from skimage.io import imread, imsave
from skimage.color import label2rgb 
import pandas as pd
import numpy as np
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

st.sidebar.subheader('Select data folder')
if st.sidebar.button('Folder Picker'):
    st.session_state['input_dir'] = filedialog.askdirectory(master=root)
st.sidebar.text_input('Selected folder:', st.session_state['input_dir'])

st.sidebar.subheader('Run model prediction')
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
            
            #Process results per class
            for c in np.unique(p['class_ids']):
                #Create names
                mask_name = f'{image_name}_masks_class-{c}.png'
                mask_path = output_dir + mask_name
                preview_name = f'{image_name}_preview_class-{c}.jpg'
                preview_path = preview_dir + preview_name

                #Get mask
                unique_class_ids = (p['class_ids'] == c).nonzero()[0]
                mask = mask_projection(p['masks'][:,:,unique_class_ids])

                #Save mask
                imsave(mask_path, mask)

                #Combine image and mask and create preview
                combined = label2rgb(mask, imread(i), bg_label = 0)
                imsave(preview_path, combined)
                nameLocation.subheader(f'Image: {image_name}')
                imageLocation.image(combined,use_column_width=True)

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
    st.sidebar.subheader('Done!')
else:
    st.sidebar.text("Click Run to process all images.")