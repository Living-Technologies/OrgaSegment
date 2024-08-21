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
        masks, labels = self.data.load_mask(image_id)
        bboxes = self.extract_bboxes(masks)
        target = {
            'masks': torch.tensor(masks),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'boxes': torch.tensor(bboxes)
        }
        return image, target



    def extract_bboxes(self, mask):
        """Compute bounding boxes from masks.

        mask: [ num_instances, height, width]. Mask pixels are either 1 or 0.

        Returns: bbox array [num_instances, (y1, x1, y2, x2)].
        """
        num_instances, height, width = mask.shape

        boxes = np.zeros((num_instances, 4), dtype=np.int32)

        row_nonzero = np.any(mask, axis=2)
        col_nonzero = np.any(mask, axis=1)

        y1 = np.argmax(row_nonzero, axis=1)
        y2 = height - np.argmax(row_nonzero[:, ::-1], axis=1) - 1
        x1 = np.argmax(col_nonzero, axis=1)
        x2 = width - np.argmax(col_nonzero[:, ::-1], axis=1) - 1

        empty_instance = ~np.any(row_nonzero, axis=1) | ~np.any(col_nonzero, axis=1)
        boxes[empty_instance] = [0, 0, 0, 0]

        boxes[:, 0] = y1
        boxes[:, 1] = x1
        boxes[:, 2] = y2 + 1
        boxes[:, 3] = x2 + 1

        return boxes


