import os
import cv2
import numpy as np
import random
import torch
import gc
from ultralytics import FastSAM
from tqdm import tqdm  # <-- Added tqdm import

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return 0.0 if union == 0 else intersection / union

# --- PATHS ---
hqseg_dir = r"D:\sems\AIP\Proj\dat\HQSeg"
fs_out_dir = r"D:\sems\AIP\Proj\dat\fsHQSeg"
os.makedirs(fs_out_dir, exist_ok=True)

# --- REFINEMENT THRESHOLDS ---
IOU_THRESHOLD = 0.90    
AREA_TOLERANCE = 0.1   

print("Loading FastSAM model...")
model = FastSAM("FastSAM-x.pt")

img_files = [f for f in os.listdir(hqseg_dir) if f.endswith('.jpg')]
print(f"Processing {len(img_files)} images...")

saved_count = 0

# --- THE FIX: Wrapped img_files in tqdm ---
for img_name in tqdm(img_files, desc="Generating Masks"):
    base_name = img_name.rsplit('.', 1)[0]
    img_path = os.path.join(hqseg_dir, img_name)
    gt_path = os.path.join(hqseg_dir, base_name + '.png')
    
    if not os.path.exists(gt_path):
        continue
        
    img = cv2.imread(img_path)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None or gt_mask is None:
        continue
        
    # Force 2D if the PNG loaded with extra channels
    if len(gt_mask.shape) > 2:
        gt_mask = gt_mask[:, :, 0]
        
    gt_bin = (gt_mask > 127).astype(np.uint8)
    gt_area = gt_bin.sum()
    
    # Drop if GT is completely empty
    if gt_area == 0:
        continue
        
    # Get a random point strictly inside the GT mask
    y_indices, x_indices = np.where(gt_bin > 0)
    rand_idx = random.randint(0, len(y_indices) - 1)
    point = [int(x_indices[rand_idx]), int(y_indices[rand_idx])]
    
    # FastSAM Prediction (VRAM Safe)
    with torch.no_grad():
        results = model(img_path, points=[point], labels=[1], verbose=False)
        
    if results[0].masks is not None and len(results[0].masks.data) > 0:
        fastsam_mask_raw = results[0].masks.data.cpu().numpy()[0]
        fs_mask = cv2.resize(fastsam_mask_raw, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        fs_bin = (fs_mask > 0.5).astype(np.uint8)
        fs_area = fs_bin.sum()
        
        # Refinement / Filtering Logic
        iou = compute_iou(fs_bin, gt_bin)
        area_diff_ratio = abs(float(fs_area) - float(gt_area)) / float(gt_area)
        
        if iou >= IOU_THRESHOLD and area_diff_ratio <= AREA_TOLERANCE:
            save_path = os.path.join(fs_out_dir, base_name + '.png')
            cv2.imwrite(save_path, fs_bin * 255)
            saved_count += 1
            
    # VRAM Cleanup
    del results
    gc.collect()
    torch.cuda.empty_cache()

print(f"\nDone! Generated and saved {saved_count} refined FastSAM masks.")