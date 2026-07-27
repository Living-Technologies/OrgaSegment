#Init logger
import logging
logger = logging.getLogger(__name__)

#Import functions
import glob
from natsort import natsorted

import pathlib

def is_image_path( pth ):
  print("checking: ", pth)
  supported = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
  return any( pth.name.endswith(sup) for sup in supported)

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
    fpath = pathlib.Path(folder)
    image_names = [ str(pth) for pth in fpath.glob("*.*") if is_image_path(pth) ]
    image_names = natsorted(image_names)
    
    imn = []
    for i in image_names:
        if image_filter in i and mask_filter not in i:
            imn.append(i)
    
    image_names = imn

    if len(image_names)==0:
        raise ValueError( 'ERROR: no jp(e)g or png images in folder %s'%folder )
    
    return image_names
