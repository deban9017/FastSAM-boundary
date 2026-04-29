import os
import cv2
cv2.setNumThreads(0)  # Prevent OpenCV deadlocks during multithreading
import numpy as np
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

# Base directory relative to code_nn
base_dir = '.' 
splits = ['test_masks']

# Thresholds
IOU_THRESHOLD = 0.85 
AREA_TOLERANCE = 0.12 # FastSAM area must be within +-12% of GT area

# High worker count because this is heavily Disk I/O bound
MAX_WORKERS = 32 

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
    
    def process_mask(mask_name):
        fs_path = os.path.join(fs_dir, mask_name)
        gt_path = os.path.join(gt_dir, mask_name)
        
        # Load as grayscale
        fs_mask = cv2.imread(fs_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if fs_mask is None or gt_mask is None:
            return 0 # Return 0 trashed
            
        # Binarize
        fs_bin = fs_mask > 127
        gt_bin = gt_mask > 127
        
        # Cast to float to prevent numpy uint8 subtraction overflow
        fs_area = float(fs_bin.sum())
        gt_area = float(gt_bin.sum())
        
        # Safety catch: if GT mask is completely empty (0 pixels), trash it
        if gt_area == 0:
            shutil.move(fs_path, os.path.join(trash_fs_dir, mask_name))
            shutil.move(gt_path, os.path.join(trash_gt_dir, mask_name))
            return 1
            
        iou = compute_iou(fs_bin, gt_bin)
        
        # Calculate how much FastSAM deviates from GT area
        area_diff_ratio = abs(fs_area - gt_area) / gt_area
        
        # Discard if IoU is too low OR area difference is strictly > 12%
        if iou < IOU_THRESHOLD or area_diff_ratio > AREA_TOLERANCE:
            shutil.move(fs_path, os.path.join(trash_fs_dir, mask_name))
            shutil.move(gt_path, os.path.join(trash_gt_dir, mask_name))
            return 1
            
        return 0

    # Execute Multithreading
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map returns the 1s and 0s from process_mask, sum them to get total trashed
        results = list(tqdm(executor.map(process_mask, mask_files), total=len(mask_files), desc=f"Refining {split}"))
        trash_count = sum(results)
        
    print(f" -> Moved {trash_count} bad pairs to {split}/trash/")

print("Dataset cleanup done! All FastSAM blowouts have been trashed.")













# import os
# import cv2
# import numpy as np
# import shutil

# def compute_iou(mask1, mask2):
#     intersection = np.logical_and(mask1, mask2).sum()
#     union = np.logical_or(mask1, mask2).sum()
#     if union == 0:
#         return 0.0
#     return intersection / union

# # Base directory relative to code_nn
# base_dir = '.' 
# splits = ['test_masks']

# # Thresholds
# IOU_THRESHOLD = 0.85 
# AREA_TOLERANCE = 0.12 # FastSAM area must be within +-12% of GT area

# for split in splits:
#     fs_dir = os.path.join(base_dir, split, 'fastsam')
#     gt_dir = os.path.join(base_dir, split, 'gt')
    
#     # Create trash folders
#     trash_fs_dir = os.path.join(base_dir, split, 'trash', 'fastsam')
#     trash_gt_dir = os.path.join(base_dir, split, 'trash', 'gt')
#     os.makedirs(trash_fs_dir, exist_ok=True)
#     os.makedirs(trash_gt_dir, exist_ok=True)
    
#     if not os.path.exists(fs_dir):
#         continue
        
#     mask_files = [f for f in os.listdir(fs_dir) if f.endswith('.png')]
#     print(f"Scanning {split} - found {len(mask_files)} masks.")
    
#     trash_count = 0
    
#     for mask_name in mask_files:
#         fs_path = os.path.join(fs_dir, mask_name)
#         gt_path = os.path.join(gt_dir, mask_name)
        
#         # Load as grayscale
#         fs_mask = cv2.imread(fs_path, cv2.IMREAD_GRAYSCALE)
#         gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
#         if fs_mask is None or gt_mask is None:
#             continue
            
#         # Binarize
#         fs_bin = fs_mask > 127
#         gt_bin = gt_mask > 127
        
#         fs_area = fs_bin.sum()
#         gt_area = gt_bin.sum()
        
#         # Safety catch: if GT mask is completely empty (0 pixels), trash it
#         if gt_area == 0:
#             shutil.move(fs_path, os.path.join(trash_fs_dir, mask_name))
#             shutil.move(gt_path, os.path.join(trash_gt_dir, mask_name))
#             trash_count += 1
#             continue
            
#         iou = compute_iou(fs_bin, gt_bin)
        
#         # Calculate how much FastSAM deviates from GT area
#         area_diff_ratio = abs(fs_area - gt_area) / gt_area
        
#         # Discard if IoU is too low OR area difference is strictly > 12%
#         if iou < IOU_THRESHOLD or area_diff_ratio > AREA_TOLERANCE:
#             shutil.move(fs_path, os.path.join(trash_fs_dir, mask_name))
#             shutil.move(gt_path, os.path.join(trash_gt_dir, mask_name))
#             trash_count += 1
            
#     print(f" -> Moved {trash_count} bad pairs to {split}/trash/")

# print("Dataset cleanup done! All FastSAM blowouts have been trashed.")