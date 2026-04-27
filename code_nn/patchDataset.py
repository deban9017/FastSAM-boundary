import os
import cv2
import numpy as np
import random
import shutil
import math

# Decisions we took for this patch dataset creation:
# 1. 50-50 Area: We only center patches on the GT boundary and ensure GT mask occupies 35% to 65% of the patch.
# 2. Strict IoU: The FastSAM patch must have >= 0.95 IoU with the GT patch.
# 3. Strict Area: The FastSAM patch mask pixel count must be within +-5% of the GT patch mask.
# 4. Test Set: Randomly moving 10 source images (and their masks) from validation to a new test set.
# 5. Spatial Diversity: Patches must be at least 100 pixels apart from each other.
# 6. No Padding: If a patch hits the image border, the crop window slides inwards to stay strictly inside the image.

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

base_dir = ''
out_base = os.path.join(base_dir, 'dat_patch')

# 1. Setup all directories
splits = ['train', 'val', 'test']
for s in splits:
    os.makedirs(os.path.join(out_base, s), exist_ok=True)
    os.makedirs(os.path.join(out_base, f'{s}_masks', 'fastsam'), exist_ok=True)
    os.makedirs(os.path.join(out_base, f'{s}_masks', 'gt'), exist_ok=True)

# 2. Pick 10 random images from val for our custom test set
val_fastsam_dir = os.path.join(base_dir, 'val_masks', 'fastsam')
if os.path.exists(val_fastsam_dir):
    val_masks = [f for f in os.listdir(val_fastsam_dir) if f.endswith('.png')]
    val_base_names = list(set([f.split('_')[0] + '_' + f.split('_')[1] for f in val_masks]))
    test_base_names = set(random.sample(val_base_names, min(10, len(val_base_names))))
    print(f"Selected {len(test_base_names)} images for the test set.")
else:
    test_base_names = set()

# 3. Process Train and Val (and split Val into Test)
process_dirs = [('train', 'train'), ('val', 'val')]
crop_size = 256
min_distance = 100 # Minimum pixels between patch centers to ensure diversity

for src_split, dest_default in process_dirs:
    fs_dir = os.path.join(base_dir, f'{src_split}_masks', 'fastsam')
    gt_dir = os.path.join(base_dir, f'{src_split}_masks', 'gt')
    img_dir = os.path.join(base_dir, src_split)
    
    if not os.path.exists(fs_dir):
        continue
        
    mask_files = [f for f in os.listdir(fs_dir) if f.endswith('.png')]
    print(f"Extracting diverse, unpadded patches from {src_split}...")
    
    for mask_name in mask_files:
        parts = mask_name.split('_')
        base_img_name = f"{parts[0]}_{parts[1]}"
        img_name = f"{base_img_name}.jpg"
        
        if src_split == 'val' and base_img_name in test_base_names:
            target_split = 'test'
        else:
            target_split = dest_default
            
        img_path = os.path.join(img_dir, img_name)
        fs_path = os.path.join(fs_dir, mask_name)
        gt_path = os.path.join(gt_dir, mask_name)
        
        if not os.path.exists(img_path) or not os.path.exists(gt_path):
            continue
            
        img = cv2.imread(img_path)
        fs_mask = cv2.imread(fs_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or fs_mask is None or gt_mask is None:
            continue
            
        h_img, w_img = img.shape[:2]
        
        # Skip image entirely if it's smaller than the crop size (rare, but safety first)
        if h_img < crop_size or w_img < crop_size:
            continue
            
        fs_bin = fs_mask > 127
        gt_bin = gt_mask > 127
        
        # Find GT boundary
        kernel = np.ones((5,5), np.uint8)
        boundary = cv2.dilate(gt_mask, kernel, iterations=1) - cv2.erode(gt_mask, kernel, iterations=1)
        by, bx = np.where(boundary > 0)
        
        if len(by) == 0:
            continue
            
        # Zip and shuffle boundary points for random, diverse selection
        boundary_points = list(zip(by, bx))
        random.shuffle(boundary_points)
            
        patches_found = 0
        selected_centers = []
        
        for (cy, cx) in boundary_points:
            if patches_found >= 4:
                break
                
            # --- Spatial Diversity Check ---
            too_close = False
            for (scy, scx) in selected_centers:
                if math.hypot(cy - scy, cx - scx) < min_distance:
                    too_close = True
                    break
            if too_close:
                continue
                
            # --- Anti-Padding Shifting Logic ---
            half = crop_size // 2
            y1, y2 = cy - half, cy + half
            x1, x2 = cx - half, cx + half
            
            # Slide window if it goes out of bounds
            if y1 < 0:
                y1, y2 = 0, crop_size
            if y2 > h_img:
                y2, y1 = h_img, h_img - crop_size
                
            if x1 < 0:
                x1, x2 = 0, crop_size
            if x2 > w_img:
                x2, x1 = w_img, w_img - crop_size
                
            img_crop = img[y1:y2, x1:x2]
            fs_crop = fs_bin[y1:y2, x1:x2]
            gt_crop = gt_bin[y1:y2, x1:x2]
            
            # --- Strict Checks ---
            gt_area = gt_crop.sum()
            total_area = crop_size * crop_size
            gt_ratio = gt_area / total_area
            if gt_ratio < 0.35 or gt_ratio > 0.65:
                continue
                
            iou = compute_iou(fs_crop, gt_crop)
            if iou < 0.95:
                continue
                
            fs_area = fs_crop.sum()
            if gt_area == 0:
                continue
            area_diff = abs(fs_area - gt_area) / gt_area
            if area_diff > 0.05:
                continue
                
            # Passed all checks!
            selected_centers.append((cy, cx))
            patches_found += 1
            patch_suffix = str(patches_found) 
            
            base_no_ext = mask_name.split('.')[0]
            save_mask_name = f"{base_no_ext}{patch_suffix}.png"
            save_img_name = f"{base_no_ext}{patch_suffix}.jpg"
            
            cv2.imwrite(os.path.join(out_base, target_split, save_img_name), img_crop)
            cv2.imwrite(os.path.join(out_base, f'{target_split}_masks', 'fastsam', save_mask_name), (fs_crop * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(out_base, f'{target_split}_masks', 'gt', save_mask_name), (gt_crop * 255).astype(np.uint8))

print("Patch dataset generation complete! Diverse patches with zero black padding.")