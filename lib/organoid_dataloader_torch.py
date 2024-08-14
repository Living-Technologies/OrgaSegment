import torch
import numpy as np
import math
class OrganoidDataset_torch(torch.utils.data.Dataset):
    def __init__(self,dataset,batch_size,device):

        self.data = dataset
        self.batch_size = batch_size
        self.images_counter = 0
        self.device = device
        self.n_images = len(self.data.image_ids)

    def __len__(self):
        return math.ceil(len(self.data.image_ids)/4)

    def __getitem__(self, index):
        images = []
        targets = []
        for _ in range(self.batch_size):
            image,target = self.load_image(self.images_counter)
            images.append(image)
            targets.append(target)
            self.images_counter += 1
            if self.images_counter == self.n_images:
                break
        return images,targets
    def load_image(self,image_id):
        image = np.asarray(self.data.load_image(image_id))
        image = np.transpose(image, (2, 0, 1))
        v_min, v_max = np.min(image), np.max(image)
        new_min, new_max = 0, 1
        image = (image - v_min) / (v_max - v_min) * (new_max - new_min) + new_min
        image = torch.tensor(image, dtype=torch.float32,device=self.device)

        masks, labels = self.data.load_mask(image_id)
        bboxes = self.extract_bboxes(masks)
        target = {'masks': torch.tensor(masks, device=self.device),
                   'labels': torch.tensor(labels, dtype=torch.int64, device=self.device),
                   'boxes': torch.tensor(bboxes, device=self.device)}

        return image,target

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