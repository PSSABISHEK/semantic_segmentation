import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from test_customDataset import TestMapillaryDataset
from torch.utils.data import DataLoader
from utils import (
    save_plots,
    load_checkpoint,
    save_test_preds_as_imgs
)

# MODELS
from models.model_a.unet.unet_model import UNet
# from torchvision.models.segmentation import fcn_resnet101

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 1
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TEST_IMG_DIR = "dataset/testing/images"

def main():

    model = UNet(n_channels=3, n_classes=66).to(DEVICE)
    # model = fcn_resnet50(pretrained=True, num_classes=66)

    test_transforms = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )

    def get_test_loader(test_dir, test_transform, batch_size, num_workers=2, pin_memory=True):
        
        test_ds = TestMapillaryDataset(
            img_dir=test_dir,
            transform=test_transform,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False
        )

        return test_loader

    test_loader = get_test_loader(TEST_IMG_DIR, test_transforms, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)

    # save_plots()

    if LOAD_MODEL:
            load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    save_test_preds_as_imgs(
        test_loader, model, folder="test_images/", device=DEVICE
    )

if __name__ == "__main__":
    main()


