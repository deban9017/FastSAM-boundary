import os
import cv2
import numpy as np
import shutil

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

# Base directory relative to code_nn
base_dir = '.' 
splits = ['train_masks', 'val_masks']

# Thresholds
IOU_THRESHOLD = 0.85 
AREA_TOLERANCE = 0.12 # FastSAM area must be within +-12% of GT area

for split in splits:
    fs_dir = os.path.join(base_dir, split, 'fastsam')
    gt_dir = os.path.join(base_dir, split, 'gt')
    
    # Create trash folders
    trash_fs_dir = os.path.join(base_dir, split, 'trash', 'fastsam')
    trash_gt_dir = os.path.join(base_dir, split, 'trash', 'gt')
    os.makedirs(trash_fs_dir, exist_ok=True)
    os.makedirs(trash_gt_dir, exist_ok=True)
    
    if not os.path.exists(fs_dir):
        continue
        
    mask_files = [f for f in os.listdir(fs_dir) if f.endswith('.png')]
    print(f"Scanning {split} - found {len(mask_files)} masks.")
    
    trash_count = 0
    
    for mask_name in mask_files:
        fs_path = os.path.join(fs_dir, mask_name)
        gt_path = os.path.join(gt_dir, mask_name)
        
        # Load as grayscale
        fs_mask = cv2.imread(fs_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if fs_mask is None or gt_mask is None:
            continue
            
        # Binarize
        fs_bin = fs_mask > 127
        gt_bin = gt_mask > 127
        
        fs_area = fs_bin.sum()
        gt_area = gt_bin.sum()
        
        # Safety catch: if GT mask is completely empty (0 pixels), trash it
        if gt_area == 0:
            shutil.move(fs_path, os.path.join(trash_fs_dir, mask_name))
            shutil.move(gt_path, os.path.join(trash_gt_dir, mask_name))
            trash_count += 1
            continue
            
        iou = compute_iou(fs_bin, gt_bin)
        
        # Calculate how much FastSAM deviates from GT area
        area_diff_ratio = abs(fs_area - gt_area) / gt_area
        
        # Discard if IoU is too low OR area difference is strictly > 12%
        if iou < IOU_THRESHOLD or area_diff_ratio > AREA_TOLERANCE:
            shutil.move(fs_path, os.path.join(trash_fs_dir, mask_name))
            shutil.move(gt_path, os.path.join(trash_gt_dir, mask_name))
            trash_count += 1
            
    print(f" -> Moved {trash_count} bad pairs to {split}/trash/")

print("Dataset cleanup done! All FastSAM blowouts have been trashed.")