#Init logger
import logging
logger = logging.getLogger(__name__)

#Import
import numpy as np

def mask_projection(mask_3D):
    """
    Create projection (2 dimensions) of masks with dimensions 3
    @params:
      mask_3D::[numpy.array] - Numpy array with 3 dimensions (y, x, mask)
    @returns:
      mask::[numpy.array] - Numpy array with 2 dimenstion (y, x). Where 0=No masks and > 0 are masks.
    """
    #Create new empty mask
    mask =  np.zeros((mask_3D.shape[0], mask_3D.shape[1]), np.uint8)

    #Process predictions
    for count, l in enumerate(range(mask_3D.shape[2])):
        #Get mask information
        msk = mask_3D[:,:,l].astype(np.uint8)
        num = l + 1
        msk = np.where(msk != 0, num, msk)
        if count == 0:
            mask = msk
        else:
            mask = np.maximum(mask, msk) #Combine previous mask with new mask
    
    return mask

def config_to_dict(config):
  configDict = {}
  for a in dir(config):
    if not a.startswith("__") and not a == "display":
      configDict[a] = getattr(config, a)
  return configDict