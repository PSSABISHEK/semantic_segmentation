import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset

# temp_mask = cv2.imread("dataset/training/labels/__CRyFzoDOXn6unQ6a3DnQ.png")
# labels, count = np.unique(temp_mask[:, :, 0], return_counts = True)
# print("Labels are : ", labels, "Counts are: ", count)

class MapillaryDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(img_dir)

    def __len__(self):
        # USE TO TEST YOU MODEL FOR INITAL RUN
        return 16
        # USE FOR PROPER TRAINING
        # return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))
        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))

        if self.transform is not None:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation['image']
            mask = augmentation['mask']
        
        return image, mask