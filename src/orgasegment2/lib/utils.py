#Init logger
import logging
logger = logging.getLogger(__name__)

#Import
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import random
import io
import pathlib
from matplotlib import patches
try:
    from PIL import Image as pil_image

    try:
        pil_image_resampling = pil_image.Resampling
    except AttributeError:
        pil_image_resampling = pil_image
except ImportError:
    pil_image = None
    pil_image_resampling = None

if pil_image_resampling is not None:
    _PIL_INTERPOLATION_METHODS = {
        "nearest": pil_image_resampling.NEAREST,
        "bilinear": pil_image_resampling.BILINEAR,
        "bicubic": pil_image_resampling.BICUBIC,
        "hamming": pil_image_resampling.HAMMING,
        "box": pil_image_resampling.BOX,
        "lanczos": pil_image_resampling.LANCZOS,
    }


def filter_masks_by_label(mask_image, class_labels, target_label):
    """
    Filters a mask image to retain only the masks corresponding to a target label.

    Parameters:
    - mask_image (np.ndarray): A 2D numpy array where each unique integer value represents a mask.
    - class_labels (np.ndarray): A 1D array of class labels corresponding to the masks in `mask_image`.
                                 The index of a class label corresponds to the integer mask value in `mask_image`.
    - target_label (int): The target label to filter for.

    Returns:
    - np.ndarray: A 2D numpy array with only the masks of the target label retained, others set to 0.
    """
    # Ensure that the mask values align with indices in the class_labels array
    if mask_image.max() >= len(class_labels):
        raise ValueError("mask_image contains values not present in class_labels indices")

    # Get mask values corresponding to the target label
    target_mask_values = np.where(class_labels == target_label)[0]

    # Create a boolean mask to filter the image
    filtered_image = np.isin(mask_image, target_mask_values) * mask_image

    return filtered_image


def config_to_dict(config):
  configDict = {}
  for a in dir(config):
    if not a.startswith("__") and not a == "display":
      configDict[a] = getattr(config, a)
  return configDict

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

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
        return
    else:
         assert boxes.shape[0] == np.unique(masks)[1:].shape[0] == class_ids[1:].shape[0]

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

    def apply_torchvision_mask_with_bitshifting(image, mask_image, boxes, labels, colors, scores, alpha=0.25):
        """
        Applies a single mask image with unique integer values for each mask to an image.

        Args:
        - image (numpy.ndarray): The original image.
        - mask_image (numpy.ndarray): A 2D array of shape (H, W) where each mask is represented by a unique integer value.
        - boxes (torch.Tensor): Tensor of shape (N, 4) containing the bounding boxes for each instance.
        - labels (torch.Tensor): Tensor of shape (N,) containing the class labels for each instance.
        - colors (list of tuples): List of RGB tuples for each class.
        - scores (torch.Tensor): Tensor of shape (N,) containing confidence scores for each instance.
        - alpha (float): Transparency of the mask overlay.

        Returns:
        - masked_image (numpy.ndarray): Image with masks applied.
        """
        # Convert the image to 32-bit integers to store encoded color values
        image = image.astype(np.uint32)

        H, W = image.shape[:2]  # Height and width of the image

        # Initialize the masked image with zeros (32-bit to store encoded color values)
        masked_image = np.zeros((H, W), dtype=np.uint32)

        unique_masks = np.unique(mask_image)  # Get unique mask values (excluding background 0)
        unique_masks = unique_masks[unique_masks > 0]  # Exclude background (value 0)

        for mask_id in unique_masks:
            # Create a binary mask for the current instance
            instance_mask = (mask_image == mask_id)

            # Find the corresponding instance index
            instance_index = mask_id - 1  # Assuming mask IDs start at 1 and match instance ordering

            # Skip instances with low scores
            # if scores[instance_index] <= 0.5:
            #     continue

            # Get the corresponding color for the label
            color = (np.array(colors[instance_index]) * 255).astype(np.uint32)

            # Encode the color into a 32-bit integer
            encoded_color = (color[0] << 16) | (color[1] << 8) | color[2]

            # Apply the mask to the image
            masked_image[instance_mask] = encoded_color

        # Decode the masked image back into 3 separate color channels (RGB)
        red_channel = (masked_image >> 16) & 0xFF
        green_channel = (masked_image >> 8) & 0xFF
        blue_channel = masked_image & 0xFF

        # Correctly stack the RGB channels into a single image
        max_projection_rgb = np.stack((red_channel, green_channel, blue_channel), axis=-1).astype(np.uint32)

        # Blend the result with the original image using the alpha value
        masked_image = ((1 - alpha) * image + alpha * max_projection_rgb).astype(np.uint8)

        return masked_image

    masked_image = apply_torchvision_mask_with_bitshifting(image,masks,boxes,class_ids,colors,scores,alpha=0.25)


    # from torchvision.utils import draw_segmentation_masks
    # import torch
    # image = np.transpose(image, (2, 0, 1))
    # masks = torch.tensor(masks, dtype=torch.bool).permute((2,0,1))
    # image = draw_segmentation_masks(torch.tensor(image),masks,alpha=0.25)
    # ax.imshow(image.permute(1,2,0))

    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue

        y1, y2, x1, x2 = boxes[i]
        if scores[i] >= 0.5:
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

def read_image(path,
               color_mode="rgb",
               target_size=None,
               interpolation="nearest",
               keep_aspect_ratio=False,):

    if pil_image is None:
        raise ImportError(
            "Could not import PIL.Image. The use of `load_img` requires PIL."
        )
    if isinstance(path, io.BytesIO):
        img = pil_image.open(path)
    elif isinstance(path, (pathlib.Path, bytes, str)):
        if isinstance(path, pathlib.Path):
            path = str(path.resolve())
        with open(path, "rb") as f:
            img = pil_image.open(io.BytesIO(f.read()))
    else:
        raise TypeError(
            f"path should be path-like or io.BytesIO, not {type(path)}"
        )

    if color_mode == "grayscale":
        # if image is not already an 8-bit, 16-bit or 32-bit grayscale image
        # convert it to an 8-bit grayscale image.
        img = np.asarray(img)
        if(len(img.shape) > 2):
            img = np.sum(img, axis=2)
        mx = img.max()
        mn = img.min()
        print("before normalization: ", mx, mn, img.shape)

        img = (img - mn)*255.0//(mx - mn)
        print("after: ", img.max(), img.min(), img.shape)
    elif color_mode == "rgba":
        if img.mode != "RGBA":
            img = img.convert("RGBA")
    elif color_mode == "rgb":
        if img.mode != "RGB":
            img = img.convert("RGB")
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    "Invalid interpolation method {} specified. Supported "
                    "methods are {}".format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys()),
                    )
                )
            resample = _PIL_INTERPOLATION_METHODS[interpolation]

            if keep_aspect_ratio:
                width, height = img.size
                target_width, target_height = width_height_tuple

                crop_height = (width * target_height) // target_width
                crop_width = (height * target_width) // target_height

                # Set back to input height / width
                # if crop_height / crop_width is not smaller.
                crop_height = min(height, crop_height)
                crop_width = min(width, crop_width)

                crop_box_hstart = (height - crop_height) // 2
                crop_box_wstart = (width - crop_width) // 2
                crop_box_wend = crop_box_wstart + crop_width
                crop_box_hend = crop_box_hstart + crop_height
                crop_box = [
                    crop_box_wstart,
                    crop_box_hstart,
                    crop_box_wend,
                    crop_box_hend,
                ]
                img = img.resize(width_height_tuple, resample, box=crop_box)
            else:
                img = img.resize(width_height_tuple, resample)
    return img
