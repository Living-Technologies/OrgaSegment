#Init logger
import logging
logger = logging.getLogger(__name__)

#Import functions
import os, glob
from natsort import natsorted
# from tensorflow import keras
import numpy as np
import random
# from tensorflow.keras.preprocessing.image import load_img
from mrcnn import utils
import re
import os
import io
import base64
import torch
from PIL import Image


def get_image_names(folder, mask_filter, image_filter=None):
    """
    Get all image names
    @params:
      folder::[str] - Location of the data
      mask_filter::[str] - String containing the mask filter name
      image_filter::[str] - String containing the image filter name (not required)
    @returns:
      image_names::[list] - List containng the 
    """
    image_names = []
    
    if image_filter is None:
        image_filter = ''
    
    image_names.extend(glob.glob(folder + '/*%s.png'%image_filter))
    image_names.extend(glob.glob(folder + '/*%s.jpg'%image_filter))
    image_names.extend(glob.glob(folder + '/*%s.jpeg'%image_filter))
    #image_names.extend(glob.glob(folder + '/*%s.tif'%image_filter)) currenlty only jp(e)g support
    #image_names.extend(glob.glob(folder + '/*%s.tiff'%image_filter)) currenlty only jp(e)g support
    image_names = natsorted(image_names)
    
    imn = []
    for i in image_names:
        if image_filter in i and mask_filter not in i:
            imn.append(i)
    
    image_names = imn

    if len(image_names)==0:
        raise ValueError('ERROR: no jp(e)g or png images in folder %s'%folder)
    
    return image_names
        
def get_mask_names(folder, mask_filter):
    """
    Get all mask names
    @params:
      folder::[str] - Location of the data
      mask_filters::[str] - String containing the mask filter name
      image_filter::[str] - String containing the image filter name (not required)
    @returns:
      image_names::[list] - List containng the 
    """
    mask_names = []
        
    mask_names.extend(glob.glob(folder + '/*%s.png'%mask_filter))
    mask_names.extend(glob.glob(folder + '/*%s.jpg'%mask_filter))
    mask_names.extend(glob.glob(folder + '/*%s.jpeg'%mask_filter))
    mask_names = natsorted(mask_names)

    if len(mask_names)==0:
        raise ValueError('ERROR: no masks in folder')
    
    return mask_names

def load_train_val_names(data_dir, image_filter=None, mask_filter='_masks', val_factor = 0.2):
    """
    Load train and validation data
    @params:
      train_dir::[str] - Location of the training data
      val_dir::[str] - Location of the validation data
      image_filter::[str] - String containing the image filter name (not required)
      mask_filter::[str] -  String containing the mask filter name. 
    @returns:
      train_images::[list] - List of training images
      train_labels::[list] - List of training image labels
      val_images::[list] - List of validation images
      val_labels::[list] - List of validation image labels
    """
    
    #Get data
    image_names = get_image_names(data_dir, mask_filter=mask_filter, image_filter=image_filter)
    nimg = len(image_names)
    
    label_names = get_mask_names(data_dir, mask_filter=mask_filter)
    nlabel = len(label_names)
    
    if nimg != nlabel:
        logger.error('Number of images is not equal to the number of masks in the directory. Every image should have a corresponding mask!')
        exit(1)
    
    # Split our img paths into a training and a validation set
    val_size = int(len(image_names) * val_factor)
    random.Random(1337).shuffle(image_names)
    random.Random(1337).shuffle(label_names)
    
    train_image_names = image_names[:-val_size]
    train_label_names = label_names[:-val_size]
    
    val_image_names = image_names[-val_size:]
    val_label_names = label_names[-val_size:]

    # Test images
    logger.info(f'Number of Train samples: {len(train_image_names)}')

    for image, label in zip(train_image_names, train_label_names):
        logger.debug(f'{image} | {label}')

    logger.info(f'Number of Validation samples: {len(val_image_names)}')

    for image, label in zip(val_image_names, val_label_names):
        logger.debug(f'{image} | {label}')

    return train_image_names, train_label_names, val_image_names, val_label_names

def array_to_base64(self, array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    encoded = base64.b64encode(buffer.getvalue())
    return encoded.decode('utf-8')

def base64_to_array(self, base64_string):
    decoded = base64.b64decode(base64_string)
    buffer = io.BytesIO(decoded)
    return np.load(buffer, allow_pickle=True)

class OrganoidGen(torch.utils.data.Dataset):
    """
    Helper to iterate over the data (as Numpy arrays).
    https://keras.io/examples/vision/oxford_pets_image_segmentation/
    """

    def __init__(self, batch_size, img_size, img_bit_depth, image_paths, label_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.bit = img_bit_depth
        self.image_paths = image_paths
        self.label_paths = label_paths

    def __len__(self):
        return len(self.label_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_image_paths = self.image_paths[i : i + self.batch_size]
        batch_label_paths = self.label_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype='float32')
        for j, path in enumerate(batch_image_paths):
            img = Image.open(path)
            img = np.asarray(img) / ((2 ** self.bit))
            x[j] = np.expand_dims(img, 2)
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype='uint16')
        for j, path in enumerate(batch_label_paths):
            img = Image.open(path)
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            # y[j] -= 1
        return x, y

class OrganoidDataset(utils.Dataset):
    '''
    Load organoid images and masks
    Create dataset for Mask RCNN training
    '''
    
    def load_data(self, data_dir, classes, image_filter='', mask_filter='', color_mode='rgb', img_bit_depth=None):
        '''Load data.
        count: number of images to generate.
        height, width: the size of the generated images.
        '''
        # Add classes
        for i, val in enumerate(classes):
            self.add_class('organoids', i + 1, val)

        # Get data information
        image_names = get_image_names(data_dir, mask_filter=mask_filter, image_filter=image_filter)
    
        for i in image_names:
            image_id = re.search(f'^{data_dir}(.*){image_filter}.*\..*$', i).group(1)

            masks = []
            for c in (c for c in classes if os.path.isfile(f'{data_dir}{image_id}{mask_filter}{c}.png')):
                mask = {'class': c,
                        'path': f'{data_dir}{image_id}{mask_filter}{c}.png'}
                masks.append(mask)
            self.add_image('organoids', image_id=image_id, path=i, color_mode=color_mode, img_bit_depth=img_bit_depth, masks=masks)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        img = Image.open(info['path'])
        
        #Convert to a 0-1 scale
        if info['img_bit_depth']:
            img = np.asarray(img) / ((2 ** info['img_bit_depth']))
        else:
            img = np.asarray(img)

        #Set extra axis if color mode is grayscale
        if info['color_mode'] == 'grayscale':
            img = img[..., np.newaxis]
        
        return img

    def info(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == 'organoids':
            return info
        else:
            super(self.__class__).image_reference(self, image_id)
    
    def load_mask(self, image_id):
        """
        Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        
        masks = info['masks']

        mask = None
        class_names = []
        
        #Split masks into numpy dimensions
        for i in masks:
            msk = Image.open(i['path'])
            msk = np.asarray(msk)
            for u in (u for u in np.unique(msk) if u > 0):
                m = np.where(msk == u, 1, 0)
                if not isinstance(mask, np.ndarray):
                    mask = m
                else:
                    mask = np.dstack((mask, m))
                class_names.append(i['class'])
            
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(i) for i in class_names])
        return mask.astype(bool), class_ids.astype(np.int32)
