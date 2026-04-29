import os
import cv2
cv2.setNumThreads(0)  # <-- FIX 1: Prevents OpenCV deadlock
import json
import numpy as np
import random
import torch
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from ultralytics import FastSAM

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

print("Loading FastSAM model...")
model = FastSAM("FastSAM-x.pt")

gpu_lock = threading.Lock()

suffixes = ['a', 'b', 'c']
splits = ['test']
base_dir = '.' 

# --- FIX 2: Dropped to 10 to prevent System RAM from overloading ---
MAX_WORKERS = 10 

for split in splits:
    img_dir = os.path.join(base_dir, split)
    fs_out_dir = os.path.join(base_dir, f'{split}_masks', 'fastsam')
    gt_out_dir = os.path.join(base_dir, f'{split}_masks', 'gt')
    os.makedirs(fs_out_dir, exist_ok=True)
    os.makedirs(gt_out_dir, exist_ok=True)
    
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    # Safely sort just in case an image is named weirdly
    try:
        img_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    except Exception:
        img_files.sort()
    
    print(f"Processing {split} folder ({len(img_files)} images)...")
    
    def process_image(img_name):
        base_name = img_name.split('.')[0]
        img_path = os.path.join(img_dir, img_name)
        json_path = os.path.join(img_dir, base_name + '.json')
        
        if not os.path.exists(json_path):
            return
            
        img = cv2.imread(img_path)
        if img is None:
            return
        shape = img.shape[:2]
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            annotations = data.get('annotations', [])
            if not annotations:
                return
            
            all_objects = []
            for ann in annotations:
                seg = ann['segmentation']
                
                if isinstance(seg, dict):
                    h, w = seg['size']
                    counts = seg['counts']
                    if isinstance(counts, list):
                        counts_arr = np.array(counts, dtype=np.int32)
                        ends = counts_arr.cumsum()
                        starts = ends - counts_arr
                        mask_1d = np.zeros(ends[-1], dtype=np.uint8)
                        for s, e in zip(starts[1::2], ends[1::2]):
                            mask_1d[s:e] = 1
                        seg_mask = mask_1d.reshape((w, h)).T
                    else:
                        from pycocotools import mask as maskUtils
                        seg_mask = maskUtils.decode(seg)
                else:
                    seg_mask = np.array(seg, dtype=np.uint8)
                
                area = np.sum(seg_mask)
                if area > 0:
                    all_objects.append({'mask': seg_mask, 'area': area})
            
            if not all_objects:
                return
                
            all_objects.sort(key=lambda x: x['area'])
            
            n = len(all_objects)
            if n >= 3:
                target_indices = [n // 4, n // 2, (3 * n) // 4]
            else:
                target_indices = list(range(n))
                
            target_indices = list(set(target_indices)) 
            
            saved_fs_masks = []
            save_count = 0
            
            for idx in target_indices:
                if save_count >= 3:
                    break
                    
                target_obj = all_objects[idx]['mask']
                
                y_indices, x_indices = np.where(target_obj > 0)
                if len(y_indices) == 0:
                    continue
                    
                rand_idx = random.randint(0, len(y_indices) - 1)
                point = [int(x_indices[rand_idx]), int(y_indices[rand_idx])]
                
                with gpu_lock:
                    with torch.no_grad():
                        results = model(img_path, points=[point], labels=[1], verbose=False)
                    
                    if results[0].masks is not None and len(results[0].masks.data) > 0:
                        fastsam_mask_raw = results[0].masks.data.cpu().numpy()[0]
                    else:
                        fastsam_mask_raw = None
                        
                    del results
                    torch.cuda.empty_cache()

                if fastsam_mask_raw is None:
                    continue

                fs_mask = cv2.resize(fastsam_mask_raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
                fs_mask_bin = (fs_mask > 0.5).astype(np.uint8)
                
                is_duplicate = False
                for saved_mask in saved_fs_masks:
                    if compute_iou(fs_mask_bin, saved_mask) > 0.90: 
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    continue
                    
                max_iou = 0
                best_gt_mask = None
                
                for obj in all_objects:
                    iou = compute_iou(fs_mask_bin, obj['mask'])
                    if iou > max_iou:
                        max_iou = iou
                        best_gt_mask = obj['mask']
                
                if best_gt_mask is not None and max_iou > 0.05:
                    saved_fs_masks.append(fs_mask_bin)
                    
                    final_fs = fs_mask_bin * 255
                    final_gt = (best_gt_mask > 0).astype(np.uint8) * 255
                    
                    save_name = f"{base_name}_{suffixes[save_count]}.png"
                    cv2.imwrite(os.path.join(fs_out_dir, save_name), final_fs)
                    cv2.imwrite(os.path.join(gt_out_dir, save_name), final_gt)
                    
                    save_count += 1
                        
        except Exception as e:
            # --- FIX 3: Actually print the error so we aren't debugging blindly ---
            print(f"\nThread crashed on {img_name} -> Error: {e}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(executor.map(process_image, img_files), total=len(img_files), desc="Generating Masks"))
        
    gc.collect()

print("Dataset generated with Max-IoU alignment and deduplication. VRAM safe!")





# import os
# import cv2
# import json
# import numpy as np
# import random
# import torch
# import gc
# from ultralytics import FastSAM

# def compute_iou(mask1, mask2):
#     intersection = np.logical_and(mask1, mask2).sum()
#     union = np.logical_or(mask1, mask2).sum()
#     if union == 0:
#         return 0.0
#     return intersection / union

# print("Loading FastSAM model...")
# model = FastSAM("FastSAM-x.pt")

# suffixes = ['a', 'b', 'c']
# splits = ['test']
# base_dir = '.' 

# for split in splits:
#     img_dir = os.path.join(base_dir, split)
    
#     fs_out_dir = os.path.join(base_dir, f'{split}_masks', 'fastsam')
#     gt_out_dir = os.path.join(base_dir, f'{split}_masks', 'gt')
#     os.makedirs(fs_out_dir, exist_ok=True)
#     os.makedirs(gt_out_dir, exist_ok=True)
    
#     img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
#     img_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
#     print(f"Processing {split} folder ({len(img_files)} images)...")
    
#     for img_name in img_files:
#         base_name = img_name.split('.')[0]
#         img_path = os.path.join(img_dir, img_name)
#         json_path = os.path.join(img_dir, base_name + '.json')
        
#         if not os.path.exists(json_path):
#             continue
            
#         img = cv2.imread(img_path)
#         if img is None:
#             continue
#         shape = img.shape[:2]
        
#         try:
#             with open(json_path, 'r') as f:
#                 data = json.load(f)
                
#             annotations = data.get('annotations', [])
#             if not annotations:
#                 continue
            
#             # 1. Decode all masks and calculate areas
#             all_objects = []
#             for ann in annotations:
#                 seg = ann['segmentation']
                
#                 if isinstance(seg, dict):
#                     h, w = seg['size']
#                     counts = seg['counts']
#                     if isinstance(counts, list):
#                         mask_1d = np.zeros(h * w, dtype=np.uint8)
#                         pos = 0
#                         for j, count in enumerate(counts):
#                             if j % 2 == 1: 
#                                 mask_1d[pos : pos + count] = 1
#                             pos += count
#                         seg_mask = mask_1d.reshape((w, h)).T
#                     else:
#                         from pycocotools import mask as maskUtils
#                         seg_mask = maskUtils.decode(seg)
#                 else:
#                     seg_mask = np.array(seg, dtype=np.uint8)
                
#                 area = np.sum(seg_mask)
#                 if area > 0:
#                     all_objects.append({'mask': seg_mask, 'area': area})
            
#             if not all_objects:
#                 continue
                
#             # 2. Sort by area
#             all_objects.sort(key=lambda x: x['area'])
            
#             # 3. Pick 3 objects: roughly 25%, 50%, 75% percentiles
#             n = len(all_objects)
#             if n >= 3:
#                 target_indices = [n // 4, n // 2, (3 * n) // 4]
#             else:
#                 target_indices = list(range(n))
                
#             target_indices = list(set(target_indices)) 
            
#             saved_fs_masks = []
#             save_count = 0
            
#             for idx in target_indices:
#                 if save_count >= 3:
#                     break
                    
#                 target_obj = all_objects[idx]['mask']
                
#                 y_indices, x_indices = np.where(target_obj > 0)
#                 if len(y_indices) == 0:
#                     continue
                    
#                 rand_idx = random.randint(0, len(y_indices) - 1)
#                 point = [int(x_indices[rand_idx]), int(y_indices[rand_idx])]
                
#                 # --- MEMORY FIX: Run FastSAM without tracking gradients ---
#                 with torch.no_grad():
#                     results = model(img_path, points=[point], labels=[1], verbose=False)
                
#                 if results[0].masks is not None and len(results[0].masks.data) > 0:
#                     fastsam_mask_raw = results[0].masks.data.cpu().numpy()[0]
#                     fs_mask = cv2.resize(fastsam_mask_raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
#                     fs_mask_bin = (fs_mask > 0.5).astype(np.uint8)
                    
#                     # 4. Deduplication Check
#                     is_duplicate = False
#                     for saved_mask in saved_fs_masks:
#                         if compute_iou(fs_mask_bin, saved_mask) > 0.90: 
#                             is_duplicate = True
#                             break
                    
#                     if is_duplicate:
#                         del results  # Free memory before continuing
#                         continue
                        
#                     # 5. Find max IoU with ALL ground truth masks
#                     max_iou = 0
#                     best_gt_mask = None
                    
#                     for obj in all_objects:
#                         iou = compute_iou(fs_mask_bin, obj['mask'])
#                         if iou > max_iou:
#                             max_iou = iou
#                             best_gt_mask = obj['mask']
                    
#                     # Save it
#                     if best_gt_mask is not None and max_iou > 0.05:
#                         saved_fs_masks.append(fs_mask_bin)
                        
#                         final_fs = fs_mask_bin * 255
#                         final_gt = (best_gt_mask > 0).astype(np.uint8) * 255
                        
#                         save_name = f"{base_name}_{suffixes[save_count]}.png"
#                         cv2.imwrite(os.path.join(fs_out_dir, save_name), final_fs)
#                         cv2.imwrite(os.path.join(gt_out_dir, save_name), final_gt)
                        
#                         save_count += 1
                
#                 # --- MEMORY FIX: Delete results object to free GPU immediately ---
#                 del results
                        
#         except Exception as e:
#             print(f"Skipping {img_name} due to error: {e}")

#         # --- MEMORY FIX: Force garbage collection and empty CUDA cache after every image ---
#         gc.collect()
#         torch.cuda.empty_cache()

# print("Dataset generated with Max-IoU alignment and deduplication. VRAM safe!")