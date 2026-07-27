#!/usr/bin/env python3

version="2.0.0"

from orgasegment2.lib import get_image_names, display_preview, filter_masks_by_label, read_image
from scipy.ndimage import find_objects

import pandas as pd
import numpy as np
import trackpy as tp
from cellpose import models
import pathlib
import os
import traceback
#Import other packages
from skimage.io import imsave
import re



#st.session_state['model_config'].COLOR_MODE)
#st.session_state['model_config'].CLASSES
RESULTS_CSV="results.csv"
TRACKED_CSV="tracked.csv"

def useGpu():
    return "USE_GPU" in os.environ and os.environ["USE_GPU"]

def predict(model_path, input_directory, output_directory, color_model='grayscale', classes=["organoid"], preview_callback=None):
    """
        Makes predictions and previews for all of the images in the input directory. Creates a results.csv with object
        prediction results.
    """
    image_path = pathlib.Path(input_directory)
    images = get_image_names(str(image_path), "mask")
    output_path = pathlib.Path(output_directory)
    if not output_path.exists():
        output_path.mkdir()
    preview_dir=pathlib.Path(output_path, "preview")
    if not preview_dir.exists():
        preview_dir.mkdir()
    print("model path expected:", model_path)
    model = models.CellposeModel(pretrained_model=model_path, gpu=useGpu())

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
            img_path = pathlib.Path(i)
            image_name = img_path.name

            #Load image
            img = np.asarray( read_image(i, color_mode=color_model) )

            #this conversion is very broken!
            #if np.amax(img) >= 255:
            #    img = ((img - np.amax(img)) * 255).astype('uint8')

            #if color_model == 'grayscale':
            #    img = img[..., np.newaxis]
            print("grayscale modded: ", img.shape)
            #Predict organoids
            masks, flows, styles = model.eval(img,diameter=None,channels=[0,0])
            print(masks.shape)

            flow_error = flows[2]
            masks = np.asarray(masks)
            slices = find_objects(masks)

            rois = []
            print(slices)
            for region in slices:
                if region is not None:  # Check if the region exists (some labels might be missing)
                    row_slice, col_slice = region
                    rois.append([row_slice.start, row_slice.stop, col_slice.start, col_slice.stop])

            # Convert to a NumPy array
            rois = np.array(rois)

            class_ids = np.ones_like(np.unique(masks))

            scores = []
            mask_ids = np.unique(masks)[1:]  # Exclude background (0)
            for mask_id in mask_ids:
                mask_region = masks == mask_id
                mean_error = flow_error[mask_region].mean()
                scores.append(mean_error)

            #Combine image and mask and create preview
            preview_name = f'{image_name}_preview.png'
            preview_path = pathlib.Path(preview_dir, preview_name)
            print("previewing: ", preview_path)
            color = np.asarray(read_image(i, color_mode='rgb'))
            print("read a color version", color.shape);
            preview = display_preview(np.asarray(read_image(i, color_mode='rgb')),
                                      rois,
                                      masks,
                                      class_ids,
                                      classes, 
                                      scores,
                                      figsize=(40, 40))
            if preview is not None:
                preview.savefig(preview_path, format='png', dpi='figure', bbox_inches='tight', pad_inches=0)

            for c in np.unique(class_ids):
                #Create names
                mask_name = f'{image_name}_masks_class-{c}.png'
                mask_path = pathlib.Path(output_path, mask_name)

                unique_class_ids = np.unique(class_ids)
                filtered_masks = filter_masks_by_label(masks, class_ids, c)
                imsave(mask_path, filtered_masks,check_contrast=False)

                #Process predictions
                for count, l in enumerate(np.unique(filtered_masks)[1:]):
                    #Get mask information
                    binary_mask = (masks == l).astype(np.uint8)
                    size = np.sum(binary_mask)
                    #Set all information
                    info = {'image': i,
                            'mask': mask_path,
                            'name': image_name,
                            'id': l,
                            'y1': rois[count,0],
                            'x1': rois[count,2],
                            'y2': rois[count,1],
                            'x2': rois[count,3],
                            'class': class_ids[count],
                            'score': scores[count],
                            'size': size}
                    info = pd.DataFrame([info])
                    results = pd.concat([results, info], ignore_index=True)
            if preview_callback is not None:
                preview_callback( (image_count+1)/len(images), preview_path )
            
        except:
            print(traceback.format_exc())

    #Save results
    csv_path = pathlib.Path(output_path, RESULTS_CSV);
    results.to_csv(str(csv_path), index=False)

#regex = st.session_state['regex']
#search_range = st.session_state['search_range']
#memory = st.session_state['memory']
#search_range = st.session_state['search_range']

def track(output_directory, regex, search_range = 50, memory = 0):
    """
        Tracks the organoids found in the results file using trackpy with
        the additional arguments.
        
    """
    output_path = pathlib.Path(output_directory, RESULTS_CSV)    
    results = pd.read_csv( output_path )
        
    #Enrich data
    results['well'] = results['name'].apply(lambda x: re.search(regex, x).group('WELL'))
    results['t'] = results['name'].apply(lambda x: re.search(regex, x).group('T'))
    
    ## Calculate centers and track organoids over time
    results['x'] = (results['x2'] + results['x1']) / 2
    results['y'] = (results['y2'] + results['y1']) / 2
    #results = results.groupby('well').apply(tp.link, search_range=int(search_range), memory=int(memory), t_column='t').reset_index(drop=True)
    results = results.groupby('well').apply(tp.link,search_range=int(search_range), t_column='t').reset_index(drop=True)
    #Save results
    tracking_csv = pathlib.Path(output_directory, TRACKED_CSV)
    results.to_csv(tracking_csv, index=False)
