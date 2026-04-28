import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
import random

class KvasirDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augmentations=None, severity="clean"):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augmentations = augmentations
        self.severity = severity
        
        # Standard ImageNet normalization applied after all augmentations
        self.normalize = transforms.Compose([
            transforms.ToTensor(), # Converts HWC uint8 [0, 255] -> CHW float32 [0.0, 1.0]
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Load image and mask
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        
        # 1. Apply standard spatial/color augmentations natively to the base image
        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image_base = augmented['image']
            mask = augmented['mask']
        else:
            image_base = image.copy()
            
        # Fork into Siamese Clean and Dark Branches
        image_clean = image_base.copy()
        
        # 2. Apply darkness stress test to Dark Branch
        current_severity = self.severity
        if current_severity == "random_mix":
            current_severity = random.choice(["clean", "mild", "medium", "severe"])
            
        if current_severity != "clean":
            from augmentations import darken_image
            image_dark = darken_image(image_base.copy(), severity=current_severity)
        else:
            image_dark = image_base.copy()
            
        # 3. Extract darkness map from the dark branch (image is still HWC uint8 RGB)
        gray = cv2.cvtColor(image_dark, cv2.COLOR_RGB2GRAY)
        darkness_mask = 1.0 - (gray.astype(np.float32) / 255.0)
        darkness_tensor = torch.tensor(darkness_mask).unsqueeze(0) # [1, H, W]
            
        # 4. Normalize and convert images to tensor
        image_clean_tensor = self.normalize(image_clean)
        image_dark_tensor = self.normalize(image_dark)
        
        # 5. Binarize and convert mask to tensor
        mask = mask.astype(np.float32) / 255.0
        mask = np.where(mask > 0.5, 1.0, 0.0).astype(np.float32)
        mask_tensor = torch.tensor(mask).unsqueeze(0) # [1, H, W]
        
        return image_clean_tensor, image_dark_tensor, mask_tensor, darkness_tensor

def get_train_val_test_splits(img_dir, mask_dir, test_size=0.1, val_size=0.1, random_state=42):
    """Returns data paths split into train/val/test"""
    images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    assert len(images) > 0, f"No images found in {img_dir}"
    assert len(images) == len(masks), "Mismatch between number of images and masks"
    
    # First split off the test set
    train_val_img, test_img, train_val_mask, test_mask = train_test_split(
        images, masks, test_size=test_size, random_state=random_state
    )
    
    # Now split the remaining into train and validation
    val_ratio = val_size / (1.0 - test_size)
    train_img, val_img, train_mask, val_mask = train_test_split(
        train_val_img, train_val_mask, test_size=val_ratio, random_state=random_state
    )
    
    return {
        "train": (train_img, train_mask),
        "val": (val_img, val_mask),
        "test": (test_img, test_mask)
    }
