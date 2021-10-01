import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset

class TestMapillaryDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.images = os.listdir(img_dir)

    def __len__(self):
        # USE TO TEST YOU MODEL FOR INITAL RUN
        # return 16
        # USE FOR PROPER TRAINING
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        image = np.array(Image.open(img_path))

        if self.transform is not None:
            augmentation = self.transform(image=image)
            image = augmentation['image']
        
        return image