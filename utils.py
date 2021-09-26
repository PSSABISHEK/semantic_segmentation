import os
import torch
import torchvision
import numpy as np
from customDataset import MapillaryDataset
from torch.utils.data import DataLoader
from pixel_to_color_map import color_map
from PIL import Image
from torchmetrics import IoU

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    # test_dir,
    # test_maskdir,
    batch_size,
    train_transform,
    val_transform,
    # test_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = MapillaryDataset(
        img_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = MapillaryDataset(
        img_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    
    # test_ds = MapillaryDataset(
    #     img_dir=test_dir,
    #     mask_dir=test_maskdir,
    #     transform=test_transform,
    # )

    # test_loader = DataLoader(
    #     test_ds,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    #     shuffle=False
    # )

    return train_loader, val_loader #, test_loader

def check_accuracy(loader, model, device="cuda", train=True):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou_metric = []
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to('cpu')
            # DONT USE NN.SOFTMAX - RETURNS AN OBJ
            preds = torch.nn.functional.softmax(model(x), dim=1)
            preds = torch.argmax(preds, dim=1)
            preds = preds.to('cpu')
            
            iou = IoU(num_classes=66)
            iou_metric.append(iou(preds, y))
    print(f"Batch mIoU Metric: {sum(iou_metric)/len(iou_metric)}")
    if train:
        model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda", train=True
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            predictions = model(x) 
            predictions = torch.nn.functional.softmax(predictions, dim=1)
            pred_labels = torch.argmax(predictions, dim=1)
            # for i in range(pred_labels.size(0)):
            save_img = torch.from_numpy(color_map(pred_labels[0]))
            print(save_img.shape)
            save_img = save_img.permute(2, 0, 1)
            # save_img = torchvision.transforms.Resize((1024, 2048))(save_img)
            torchvision.utils.save_image(
                save_img, f"{folder}pred_{idx}_{0}.png", normalize=True
            )
        
        # torchvision.utils.save_image(y.unsqueeze(3), f"{folder}/{idx}.png")

    model.train()