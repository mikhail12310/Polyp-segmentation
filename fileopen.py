import os
import matplotlib
matplotlib.use('Agg')  # no display needed
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

image_dir = "Kvasir-SEG/images"
mask_dir  = "Kvasir-SEG/masks"

filenames = os.listdir(image_dir)
samples = random.sample(filenames, 5)

fig, axes = plt.subplots(5, 2, figsize=(8, 20))
for i, fname in enumerate(samples):
    img  = mpimg.imread(os.path.join(image_dir, fname))
    mask = mpimg.imread(os.path.join(mask_dir, fname))

    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"Image: {fname}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(mask, cmap="gray")
    axes[i, 1].set_title("Mask")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.savefig("preview.png", dpi=150)
print("Saved preview.png")