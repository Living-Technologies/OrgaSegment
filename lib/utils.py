#Init logger
import logging
logger = logging.getLogger(__name__)

#Import
import numpy as np
import matplotlib.pyplot as plt
from mrcnn.visualize import random_colors, patches, apply_mask, find_contours, Polygon

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

    masked_image = image.astype(np.uint32).copy()
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

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color, alpha=0.25)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    
    ax.imshow(masked_image.astype(np.uint8))
    return fig