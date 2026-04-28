import albumentations as A
import cv2
import numpy as np

def get_training_augmentation(img_size):
    """Standard augmentations for Kvasir-SEG (Clean)"""
    train_transform = [
        A.Resize(height=img_size[0], width=img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        # Note: Normalize usually outputs float32, keeping it strictly separate from ToTensorV2 for raw torch tensor processing later if needed
    ]
    return A.Compose(train_transform)

def get_validation_augmentation(img_size):
    """Validation/Test augmentations - Only resizing"""
    val_transform = [
         A.Resize(height=img_size[0], width=img_size[1]),
    ]
    return A.Compose(val_transform)

def darken_image(image, severity="mild"):
    """
    Simulates real-world endoscopy degradations.
    image: uint8 numpy array (H, W, C), RGB format
    Returns corrupted uint8 numpy array (H, W, C)
    """
    if severity == "clean":
        return image
        
    img = image.copy().astype(np.float32) / 255.0
    
    if severity == "mild":
        gamma = 1.5
        noise_std = 0.02
        contrast = 0.9
        vignette_str = 0.3
    elif severity == "medium":
        gamma = 2.5
        noise_std = 0.05
        contrast = 0.7
        vignette_str = 0.6
    elif severity == "severe":
        gamma = 3.5
        noise_std = 0.1
        contrast = 0.5
        vignette_str = 0.8
    else:
        return image
        
    # 1. Contrast reduction
    mean_lum = np.mean(img)
    img = (img - mean_lum) * contrast + mean_lum
    img = np.clip(img, 0, 1)
    
    # 2. Gamma darkening
    img = np.power(img, gamma)
    
    # 3. Radial vignetting and uneven illumination
    h, w = img.shape[:2]
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    radius = np.sqrt(X**2 + Y**2)
    # Falloff for vignette
    vignette_mask = 1 - vignette_str * (radius / np.max(radius))
    vignette_mask = np.clip(vignette_mask, 0, 1)
    
    # Uneven illumination gradient (simulates offset light source usually on scope tip)
    grad_mask = 1.0 - 0.2 * ((X + Y) / 2 + 1)
    grad_mask = np.clip(grad_mask, 0, 1)
    
    combined_mask = vignette_mask * grad_mask
    img = img * combined_mask[..., np.newaxis]
    
    # 4. Gaussian noise (low light image sensor noise)
    noise = np.random.normal(0, noise_std, img.shape)
    img = img + noise
    
    img = np.clip(img, 0, 1)
    return (img * 255.0).astype(np.uint8)
