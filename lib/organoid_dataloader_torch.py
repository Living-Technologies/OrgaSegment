import torch
import numpy as np
import math
import time

from matplotlib import pyplot as plt


class OrganoidDataset_torch(torch.utils.data.Dataset):
    def __init__(self,dataset,batch_size):

        self.data = dataset
        self.batch_size = batch_size
        self.images_counter = 0
        self.n_images = len(self.data.image_ids)

    def __len__(self):
        return math.ceil(len(self.data.image_ids)/4)

    def __getitem__(self, index):
        images = []
        targets = []
        for _ in range(self.batch_size):
            image, target = self.load_image(self.images_counter)

            images.append(image)
            targets.append(target)

            self.images_counter += 1
            if self.images_counter == self.n_images:
                break
        return images,targets

    def load_image(self, image_id):

        image = self.data.load_image(image_id)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        image = (image - image.min()) / (image.max() - image.min())

        masks, labels = self.data.load_mask_new(image_id)

        bboxes = self.extract_bboxes_new(masks)
        # bboxes = self.extract_bboxes(masks)

        target = {
            'masks': torch.tensor(masks),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'boxes': torch.tensor(bboxes)
        }

        return image, target

    def extract_bboxes(self,mask):
        """Compute bounding boxes from masks.
        mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

        Returns: bbox array [num_instances, (y1, x1, y2, x2)].
        """
        boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
            # Bounding box.
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, x2, y1, y2 = 0, 0, 0, 0
            boxes[i] = np.array([y1, x1, y2, x2])
        return boxes.astype(np.int32)

    def extract_bboxes_new(self, mask):
        """Compute bounding boxes from masks.

        mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

        Returns: bbox array [num_instances, (y1, x1, y2, x2)].
        """
        height, width, num_instances = mask.shape

        # Initialize the output array
        boxes = np.zeros((num_instances, 4), dtype=np.int32)

        # Compute row and column indices where masks are present
        row_nonzero = np.any(mask, axis=1)  # Shape: [height, num_instances]
        col_nonzero = np.any(mask, axis=0)  # Shape: [width, num_instances]

        # Find first and last non-zero indices in rows and columns
        y1 = np.argmax(row_nonzero, axis=0)  # First non-zero row for each instance
        y2 = height - np.argmax(row_nonzero[::-1], axis=0) - 1  # Last non-zero row
        x1 = np.argmax(col_nonzero, axis=0)  # First non-zero column
        x2 = width - np.argmax(col_nonzero[::-1], axis=0) - 1  # Last non-zero column

        # Handle cases where there are no non-zero rows or columns
        empty_instance = ~np.any(row_nonzero, axis=0) | ~np.any(col_nonzero, axis=0)
        boxes[empty_instance] = [0, 0, 0, 0]

        boxes[:, 0] = y1
        boxes[:, 1] = x1

        # Increment y2 and x2 by 1 to include the last pixel in the bounding box
        boxes[:, 2] = y2 + 1
        boxes[:, 3] = x2 + 1

        return boxes


