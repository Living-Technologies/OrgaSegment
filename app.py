import streamlit as st
import os

st.title('OrgaSegment')

st.sidebar.subheader('Select configuration')
config_path = st.sidebar.file_uploader('Select configuration')

# st.sidebar.subheader('Select data folder')
# data_path = st.sidebar.file_uploader('Select data folder')

# if config_path is not None and data_path is not None:
#     os.system(f'python predict_mrcnn.py UNKNOWN {config_path} {data_path}')