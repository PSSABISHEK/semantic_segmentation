import os
import torch
import torchvision
import numpy as np
from customDataset import MapillaryDataset
from torch.utils.data import DataLoader
from pixel_to_color_map import color_map
from PIL import Image

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
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = MapillaryDataset(
        img_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
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

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.nn.Softmax(model(x))
            # preds = (preds > 0.5).float()
            # num_correct += (preds == y).sum()
            # num_pixels += torch.numel(preds)
            # dice_score += (2 * (preds * y).sum()) / (
            #     (preds + y).sum() + 1e-8
            # )

    # print(
    #     f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    # )
    # print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            predictions = model(x) 
            predictions = torch.nn.functional.softmax(predictions, dim=1)
            pred_labels = torch.argmax(predictions, dim=1)
        for i in range(pred_labels.size(0)):
            save_img = torch.from_numpy(color_map(pred_labels[i]))
            print(save_img.shape)
            save_img = save_img.permute(2, 0, 1)
            save_img = torchvision.transforms.Resize((1024, 2048))(save_img)
            torchvision.utils.save_image(
                save_img, f"{folder}pred_{idx}_{i}.png", multiclass=True
            )
        
        # torchvision.utils.save_image(y.unsqueeze(3), f"{folder}/{idx}.png")

    model.train()