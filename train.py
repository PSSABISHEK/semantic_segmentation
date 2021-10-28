import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet101, lraspp_mobilenet_v3_large

# from models.model_a.unet.unet_model import UNet
# from models.model_b.models.fusenet_model import FuseNet
# from models.model_c.models.fcn32s import FCN32VGG
# from models.model_d.train.erfnet import Net
# from models.model_e.network.gscnn import GSCNN


from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_metircs,
    save_predictions_as_imgs,
)

# Hyperparametthoners etc.
LEARNING_RATE = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 2
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "dataset/training/images"
TRAIN_MASK_DIR = "dataset/training/labels"
VAL_IMG_DIR = "dataset/validation/images"
VAL_MASK_DIR = "dataset/validation/labels"
TEST_IMG_DIR = "dataset/testing/images"

writer = SummaryWriter()

def train_fn(loader, model, optimizer, loss_fn, scaler, curr_epoch):
    model.to(device=DEVICE)
    loop = tqdm(loader)
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)

        #forward
        with torch.cuda.amp.autocast():
            prediction = model(data)
            # loss = loss_fn(prediction, targets)
            loss = loss_fn(prediction['out'], targets)
        
        running_loss += loss.item()

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        #update tqdm
        loop.set_postfix(loss=loss.item())

    writer.add_scalar("Loss/train", running_loss/len(loader), curr_epoch)
    running_loss = 0.0
    writer.close()



def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.6, saturation=0, hue=0.5, p=0.4),
            # A.VerticalFlip(p=0.1),
            A.Normalize(
                # mean=[0.0, 0.0, 0.0],
                mean=[0.485, 0.456, 0.406],
                # std=[1.0, 1.0, 1.0],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
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

    # CHANGE LINE BELOW FOR NEW MODELS
    # model = UNet(n_channels=3, n_classes=66)

    # model = fcn_resnet50(pretrained=True, progress=True)
    # model.classifier = FCNHead(2048, 66)
    
    model = deeplabv3_resnet101(pretrained=True, progress=True)
    model.classifier = DeepLabHead(2048, 66)
    
    # model = lraspp_mobilenet_v3_large(pretrained=False, progress=False, num_classes=66).to(DEVICE)
    
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    learning_rate_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    # check_metircs(val_loader, model, loss_fn, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_metircs(val_loader, model, loss_fn, epoch, device=DEVICE)

        # Step LR
        learning_rate_scheduler.step()

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
    
    writer.close()

if __name__ == "__main__":
    main()