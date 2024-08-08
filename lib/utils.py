#Init logger
import logging
logger = logging.getLogger(__name__)

#Import
import numpy as np
import matplotlib.pyplot as plt
from mrcnn.visualize import random_colors, patches, apply_mask, find_contours, Polygon
import skimage as ski

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

def display_preview(image, boxes, masks, class_ids, class_names,
                    scores=None, figsize=(20, 20),
                    show_mask=True, show_bbox=True,
                    colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    plt.close()
    # Create plot
    fig, ax = plt.subplots(1, figsize=figsize)

    #Check class names
    if class_names[0] != 'BG':
        class_names.insert(0, 'BG')  

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')

    def apply_mask_with_bitshifting():
        ids = np.arange(1, N + 1)
        ids_multi_dim = ids[np.newaxis, np.newaxis, :]
        id_masks = masks * ids_multi_dim
        max_projection = np.max(id_masks, axis=2)

        # Encode colors into 32-bit integers
        colors_large = (np.asarray(colors) * 255).astype(np.uint32)
        encoded_colors = (colors_large[:, 0] << 16) | (colors_large[:, 1] << 8) | colors_large[:, 2]

        # Initialize the masked image with zeros (32-bit to store encoded color values)
        masked_image = np.zeros(max_projection.shape, dtype=np.uint32)

        # Apply bitwise operations to generate the masked_image
        for i in range(1, N + 1):
            mask = (max_projection == i)
            masked_image[mask] = encoded_colors[i - 1]

        # Decode the masked image back into 3 separate color channels (RGB)
        red_channel = (masked_image >> 16) & 0xFF
        green_channel = (masked_image >> 8) & 0xFF
        blue_channel = masked_image & 0xFF

        max_projection_rgb = np.stack((red_channel, green_channel, blue_channel), axis=2).astype(np.uint32)

        # Blend the result with the original image using the alpha value
        # Blend is done using bitshifts, assuming alpha =0.25 is used the operations are as follows:
        # masked image = 0.75 * image + 0.25 * max_projection_rgb
        masked_image = np.right_shift(image,2) * 3 + np.right_shift(max_projection_rgb,2)
        return masked_image


    masked_image = apply_mask_with_bitshifting()

    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='black', size=18, backgroundcolor='none')

    ax.imshow(masked_image.astype(np.uint8))
    return fig