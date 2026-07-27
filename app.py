
import streamlit as st
import configparser

#Import cellpose packages
from cellpose import models
from matplotlib import pyplot as plt

import sys

orgasegment_found = False
try:
    import orgasegment2
    orgasegment_found = True
except:
    sys.path.append("src")
    import orgasegment2
    orgasegment_found = True
finally:
    if not orgasegment_found:
        raise Exception("could not load orgasegment. Run in repository folder, or install in environment.")
#Import OrgaSegment functions
from orgasegment2.lib import get_image_names, display_preview, filter_masks_by_label, read_image

#Import other packages
from skimage.io import imsave
import pandas as pd
import numpy as np
import trackpy as tp
import re
import os
from pathlib import Path
import traceback
from scipy.ndimage import find_objects


#Get app config
config = configparser.ConfigParser()
config.sections()
config.read('./conf/appCellPose.conf')
config.sections()

if __name__=="__main__":
    print("starting from the cmd line")
    print(__file__)

# Check if 'input_dir' already exists in session_state
# If not, then initialize it
if 'model' not in st.session_state:
    st.session_state['model'] = 'None'
if 'model_path' not in st.session_state:
    st.session_state['model_path'] = 'None'
if 'input_dir' not in st.session_state:
    st.session_state['input_dir'] = 'None'



#####Create app
st.title('OrgaSegment2')
nameLocation = st.empty()
imageLocation = st.empty()
st.sidebar.header('Settings')
st.sidebar.subheader('Select model')
st.session_state['model'] = st.sidebar.selectbox('Please select model for segmentation', config.sections())
config_values = config[st.session_state['model']]
st.session_state['model_path'] = config_values['model']
st.session_state['model_path'] = st.sidebar.text_input('Selected model:', st.session_state['model_path'])

st.sidebar.subheader('Select options')
st.session_state['predict'] = st.sidebar.checkbox('Inference')
st.session_state['track'] = st.sidebar.checkbox('Track')

st.sidebar.subheader('Select data folder')

st.session_state['input_dir'] = config_values['data']
st.session_state['input_dir'] = st.sidebar.text_input('Selected folder:', st.session_state['input_dir'])

if st.session_state['track']:
    st.sidebar.subheader('Select track options')
    st.session_state['regex'] = st.sidebar.text_input('File name regex, correct if needed',  '.*(?P<WELL>[A-Z]{1}[0-9]{1,2}).*t(?P<T>[0-9]{1,2}).*')
    st.session_state['search_range'] = st.sidebar.text_input('Tracking search range in pixels, correct if needed',  '50')
    st.session_state['memory'] = st.sidebar.text_input('Memory: the maximum number of frames during which an organoid can vanisch, then reappear within the search range, and be considered the same organoid. Correct if needed',  '0')

st.sidebar.subheader('Run application')
if st.sidebar.button('Run'):

    progress_bar = st.sidebar.progress(0)
    #Get data
    #images = get_image_names(st.session_state['input_dir'], '_masks')
    #Create folders
    input_dir=os.path.join(st.session_state['input_dir'], '')
    output_dir=os.path.join(input_dir, st.session_state['model'], '')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    preview_dir=os.path.join(input_dir, st.session_state['model'], 'preview', '')
    Path(preview_dir).mkdir(parents=True, exist_ok=True)

    if st.session_state['predict']:

        model_path = st.session_state['model_path']
        input_directory = st.session_state['input_dir']
        output_dir = os.path.join(input_dir, st.session_state['model'], '')
        classes = ['organoid']
        color_model = 'grayscale'
        
        def previewCallback(progress, preview_image):
            image_name = preview_image.name
            nameLocation.subheader(f'Image: {image_name}')
            imageLocation.image(read_image(preview_image, color_mode='rgb'), use_container_width=True)
            progress_bar.progress(progress)
        
        orgasegment2.predict(model_path, input_directory, output_dir, color_model=color_model, classes=classes, preview_callback=previewCallback)

    #Track
    if st.session_state['track']:        
        regex = st.session_state['regex']
        search_range = st.session_state['search_range']
        memory = st.session_state['memory']
        
        orgasegment2.track(output_dir, regex, search_range = search_range, memory = memory)
    
    
    st.sidebar.subheader('Done!')

else:
    st.sidebar.text("Click Run to process all images.")
