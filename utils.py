from collections import deque
import os
import torch
import torchvision
import numpy as np
from customDataset import MapillaryDataset
from torch.utils.data import DataLoader
from pixel_to_color_map import color_map
from PIL import Image
from torchmetrics import IoU
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def save_plots():
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax.plot([0,1,2], [10,20,3])
    fig.savefig('metric.png')   # save the figure to file
    plt.close(fig)  

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
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

    return train_loader, val_loader

def check_metircs(loader, model, loss_fn, curr_epoch, device="cuda", train=True):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    running_loss = 0
    iou_metric = []
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.long().to(device)
            # DONT USE NN.SOFTMAX - RETURNS AN OBJ
            preds = model(x)
            # COMMENT LINE BELOW WHEN TRAINING ON UNET
            preds = preds['out']
            loss = loss_fn(preds, y)
            preds = torch.nn.functional.softmax(preds, dim=1)
            preds = torch.argmax(preds, dim=1)
            preds = preds.to('cpu')
            y = y.to('cpu')
            running_loss += loss.item()
            iou = IoU(num_classes=66)
            iou_metric.append(iou(preds, y))
    writer.add_scalar("Loss/validation", running_loss/len(loader), curr_epoch)
    writer.add_scalar("mIoU/validation", (sum(iou_metric)/len(iou_metric))*100, curr_epoch)
    print(f"Val Loss: {running_loss/len(loader)}")
    print(f"Batch mIoU Metric: {(sum(iou_metric)/len(iou_metric))*100}")
    writer.close()
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
            # COMMENT LINE BELOW WHEN TRAINING ON UNET
            predictions = predictions['out']
            predictions = torch.nn.functional.softmax(predictions, dim=1)
            pred_labels = torch.argmax(predictions, dim=1)
            # for i in range(pred_labels.size(0)):
            save_img = torch.from_numpy(color_map(pred_labels[0]))
            save_img = save_img.permute(2, 0, 1)
            save_img = torchvision.transforms.Resize((1024, 2048))(save_img)
            torchvision.utils.save_image(
                save_img, f"{folder}pred_{idx}_{0}.png", normalize=True
            )

    model.train()

def save_test_preds_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, x in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            predictions = model(x)
            # COMMENT LINE BELOW WHEN TRAINING ON UNET
            predictions = predictions['out']
            predictions = torch.nn.functional.softmax(predictions, dim=1)
            pred_labels = torch.argmax(predictions, dim=1)
            for i in range(pred_labels.size(0)):
                save_img = torch.from_numpy(color_map(pred_labels[i]))
                print(save_img.shape)
                save_img = save_img.permute(2, 0, 1)
                save_img = torchvision.transforms.Resize((1024, 2048))(save_img)
                torchvision.utils.save_image(
                    save_img, f"{folder}pred_{idx}_{i}.png", normalize=True
                )