"""
OUTPUT:

Loading ONNX C++ Runtime for CPU...

--- Benchmarking FastSAM ONNX on CPU (50 runs) ---
FastSAM CPU Bench: 100%|███████████████████████████████████████████████████████████| 50/50 [01:27<00:00,  1.76s/it]

--- Benchmarking Our Unified Model ONNX on CPU (50 runs) ---
Our Model CPU Bench: 100%|█████████████████████████████████████████████████████████| 50/50 [00:42<00:00,  1.17it/s]

======================================
Average FastSAM ONNX CPU Inference Time:   1.7599 seconds
Average Unified ONNX CPU Inference Time:   0.8556 seconds
======================================


Before concatenation, OUTPUT:

Loading ONNX C++ Runtime for CPU...

--- Benchmarking FastSAM ONNX on CPU (50 runs) ---
FastSAM CPU Bench: 100%|███████████████████████████████████████████████████████████| 50/50 [01:27<00:00,  1.75s/it]

--- Benchmarking Our Model ONNX on CPU (200 runs) ---
Our Model CPU Bench: 100%|███████████████████████████████████████████████████████| 200/200 [03:36<00:00,  1.08s/it]

======================================
Average FastSAM ONNX CPU Inference Time:   1.7531 seconds
Average Our Model ONNX CPU Inference Time: 1.0806 seconds
======================================
"""


import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

print("Loading ONNX C++ Runtime for CPU...")
# Initialize ONNX sessions using the purely C++ CPU execution provider
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Load FastSAM and our new unified pipeline ONNX models
fastsam_session = ort.InferenceSession("FastSAM-x.onnx", sess_options, providers=['CPUExecutionProvider'])
pipeline_session = ort.InferenceSession("full_pipeline_optcat5.onnx", sess_options, providers=['CPUExecutionProvider'])

img_path = r"D:\sems\AIP\Proj\acceleration\i1.jpg"
mask_path = r"D:\sems\AIP\Proj\acceleration\m1.png"

img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W, _ = img_rgb.shape

fs_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
fs_bin = (fs_mask > 127).astype(np.float32).squeeze()

# ==========================================
# 1. FASTSAM ONNX BENCHMARK
# ==========================================
# FastSAM exported at imgsz=1024, normalize 0-1, format BCHW
img_resized = cv2.resize(img_rgb, (1024, 1024))
fs_input_np = (img_resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
fs_input_name = fastsam_session.get_inputs()[0].name

NUM_RUNS_FS = 50

print(f"\n--- Benchmarking FastSAM ONNX on CPU ({NUM_RUNS_FS} runs) ---")
# Warmup
for _ in range(5):
    fastsam_session.run(None, {fs_input_name: fs_input_np})

t0 = time.time()
for _ in tqdm(range(NUM_RUNS_FS), desc="FastSAM CPU Bench"):
    fastsam_session.run(None, {fs_input_name: fs_input_np})
fs_avg_time = (time.time() - t0) / NUM_RUNS_FS

# ==========================================
# 2. PREPARE OUR MODEL INPUTS
# ==========================================
# Prepare coordinates
fs_uint8 = (fs_bin * 255).astype(np.uint8)
contours, _ = cv2.findContours(fs_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

patch_size = 256
half = patch_size // 2
stride = 150 
patch_centers = [(cnt[i][0][0], cnt[i][0][1]) for cnt in contours for i in range(0, len(cnt), stride)]

coords, img_crops, mask_crops = [], [], []
for x, y in patch_centers:
    x1, x2 = max(0, x - half), min(W, x + half)
    y1, y2 = max(0, y - half), min(H, y + half)
    if x2 - x1 < patch_size: x1, x2 = (0, patch_size) if x1 == 0 else (W - patch_size, W)
    if y2 - y1 < patch_size: y1, y2 = (0, patch_size) if y1 == 0 else (H - patch_size, H)
        
    coords.append((x1, y1, x2, y2))
    img_crops.append(img_rgb[y1:y2, x1:x2])
    mask_crops.append(fs_bin[y1:y2, x1:x2])

# Prepare Numpy batched inputs
all_imgs_np = (np.stack(img_crops).astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
all_masks_np = np.stack(mask_crops).reshape(-1, 1, patch_size, patch_size).astype(np.float32)

BATCH_SIZE = 32
NUM_RUNS_OURS = 50

# ==========================================
# 3. OUR UNIFIED MODEL ONNX BENCHMARK
# ==========================================
print(f"\n--- Benchmarking Our Unified Model ONNX on CPU ({NUM_RUNS_OURS} runs) ---")
# Warmup
for _ in range(5):
    for i in range(0, len(coords), BATCH_SIZE):
        batch_img = all_imgs_np[i:i+BATCH_SIZE]
        batch_mask = all_masks_np[i:i+BATCH_SIZE]
        _ = pipeline_session.run(['output'], {'image': batch_img, 'mask': batch_mask})[0]

t0 = time.time()
for _ in tqdm(range(NUM_RUNS_OURS), desc="Our Model CPU Bench"):
    full_prob_opt = fs_bin.copy()
    full_count_opt = np.ones((H, W), dtype=np.float32)
    
    for i in range(0, len(coords), BATCH_SIZE):
        batch_img = all_imgs_np[i:i+BATCH_SIZE]
        batch_mask = all_masks_np[i:i+BATCH_SIZE]
        
        # Run the single fused pipeline
        inputs = {'image': batch_img, 'mask': batch_mask}
        m2_out = pipeline_session.run(['output'], inputs)[0]
        
        pred_patches = np.squeeze(m2_out, axis=1)
        
        # Stitch
        for j in range(len(pred_patches)):
            x1, y1, x2, y2 = coords[i+j]
            full_prob_opt[y1:y2, x1:x2] += pred_patches[j]
            full_count_opt[y1:y2, x1:x2] += 1

our_total_time = time.time() - t0
our_avg_time = our_total_time / NUM_RUNS_OURS

print(f"\n======================================")
print(f"Average FastSAM ONNX CPU Inference Time:   {fs_avg_time:.4f} seconds")
print(f"Average Unified ONNX CPU Inference Time:   {our_avg_time:.4f} seconds")
print(f"======================================")