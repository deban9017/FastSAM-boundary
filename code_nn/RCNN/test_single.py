import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from model import IterativeBoundaryRefiner

def create_overlay(img_base, mask, mask_color=(0, 255, 0)):
    mask_bin = (mask > 0.5).astype(np.uint8)
    color_layer = np.zeros_like(img_base)
    color_layer[:] = mask_color
    alpha = mask_bin[..., None] * 0.5 
    overlay = (img_base * (1 - alpha) + color_layer * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
    return overlay

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

refiner = IterativeBoundaryRefiner().to(device)
refiner.load_state_dict(torch.load('code_nn/RCNN/refiner_epoch_9.pth')) # Adjust epoch
refiner.eval()

# Pick a random patch from the test set
test_img_dir = 'dat_patch/test'
test_fs_dir = 'dat_patch/test_masks/fastsam'
test_gt_dir = 'dat_patch/test_masks/gt'

test_imgs = [f for f in os.listdir(test_img_dir) if f.endswith('.jpg')]
img_name = random.choice(test_imgs)
mask_name = img_name.replace('.jpg', '.png')

img = cv2.imread(os.path.join(test_img_dir, img_name))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fs_mask = cv2.imread(os.path.join(test_fs_dir, mask_name), cv2.IMREAD_GRAYSCALE)
gt_mask = cv2.imread(os.path.join(test_gt_dir, mask_name), cv2.IMREAD_GRAYSCALE)

fs_bin = (fs_mask > 127).astype(np.uint8)
gt_bin = (gt_mask > 127).astype(np.uint8)

# Convert to tensor and pass to model
img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
fs_t = torch.from_numpy(fs_bin.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    refined_crop_t = refiner(img_t, fs_t, iters=4)
    refined_crop = refined_crop_t.squeeze().cpu().numpy()

refined_crop_bin = (refined_crop > 0.5).astype(np.uint8)

# Take the union to avoid inner holes
union_crop = np.logical_or(fs_bin, refined_crop_bin).astype(np.uint8)

# Plotting
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(img_rgb); axes[0].set_title("Input Patch")
axes[1].imshow(create_overlay(img_rgb, gt_bin)); axes[1].set_title("Ground Truth")
axes[2].imshow(create_overlay(img_rgb, fs_bin)); axes[2].set_title("FastSAM Output")
axes[3].imshow(create_overlay(img_rgb, union_crop)); axes[3].set_title("NN Refined Output")

for ax in axes: ax.axis('off')
plt.tight_layout()
plt.savefig('code_nn/RCNN/basic_patch_result.png', bbox_inches='tight', dpi=300)
plt.show()