#Init logger
import logging
logger = logging.getLogger(__name__)

#Import functions
import os, glob
from natsort import natsorted
from tensorflow import keras
import numpy as np
import random
from tensorflow.keras.preprocessing.image import load_img

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
    image_names.extend(glob.glob(folder + '/*%s.tif'%image_filter))
    image_names.extend(glob.glob(folder + '/*%s.tiff'%image_filter))
    image_names = natsorted(image_names)
    
    imn = []
    for i in image_names:
        if image_filter in i and mask_filter not in i:
            imn.append(i)
    
    image_names = imn

    if len(image_names)==0:
        raise ValueError('ERROR: no images in folder')
    
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
    mask_names.extend(glob.glob(folder + '/*%s.tif'%mask_filter))
    mask_names.extend(glob.glob(folder + '/*%s.tiff'%mask_filter))
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

    logger.info(f'Number of Validation samples: {len(train_image_names)}')

    for image, label in zip(val_image_names, val_label_names):
        logger.debug(f'{image} | {label}')

    return train_image_names, train_label_names, val_image_names, val_label_names


class OrganoidGen(keras.utils.Sequence):
    """
    Helper to iterate over the data (as Numpy arrays).
    https://keras.io/examples/vision/oxford_pets_image_segmentation/
    """

    def __init__(self, batch_size, img_size, image_paths, label_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.image_paths = image_paths
        self.label_paths = label_paths

    def __len__(self):
        return len(self.label_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_image_paths = self.image_paths[i : i + self.batch_size]
        batch_label_paths = self.label_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint16")
        for j, path in enumerate(batch_image_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            x[j] = np.expand_dims(img, 2)
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint16")
        for j, path in enumerate(batch_label_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            # y[j] -= 1
        return x, y

#Test image gen
# x = train_gen.__getitem__(0)

#     for i in range(len(x[0])):
#         image = x[0][i]
#         plt.imshow(image)
#         plt.savefig(f'test/{i}-image.png')

#         label = x[1][i]
#         plt.imshow(label)
#         plt.savefig(f'test/{i}-label.png')

#     print('succes')