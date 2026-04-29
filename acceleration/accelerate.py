"""
OUTPUT:

GPUs available: 1

Loading Models...

--- Running FastSAM Baseline (10 runs for average) ---
FastSAM Average Inference Time: 0.2146 seconds

Total boundary patches to process per image: 15

--- Running Unoptimized Pipeline (BS=1) for 500 iterations ---
Unopt Bench: 100%|███████████████████████████████████████████████████████████████| 500/500 [02:10<00:00,  3.84it/s]
Average Unoptimized Time per image: 0.2605 seconds

--- Running Optimized Pipeline (Batched) for 500 iterations ---
Opt Bench: 100%|█████████████████████████████████████████████████████████████████| 500/500 [02:04<00:00,  4.01it/s]
Average Batched Optimized Time per image: 0.2491 seconds
True Speedup vs Unoptimized: 1.05x

CPU timing Output:

Total boundary patches to process: 15

--- Running CPU Pipeline (Batched) for 50 iterations ---
CPU Bench: 100%|███████████████████████████████████████████████████████████████████| 50/50 [01:30<00:00,  1.82s/it]

======================================
Average CPU Inference Time: 1.8164 seconds
======================================

"""


# ===== CLEAR GPU MEMORY =====
import gc
import os
import time
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import FastSAM
from tqdm import tqdm

gc.collect()
torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("GPUs available:", torch.cuda.device_count())

# ==========================================
# 1. MODEL CLASSES
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class BoundaryPuller(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        self.mask_stem = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        self.pre_loop = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        self.looper = nn.Sequential(
            ResBlock(24, dilation=1), ResBlock(24, dilation=2),
            ResBlock(24, dilation=4), nn.Conv2d(24, 1, kernel_size=3, padding=1)
        )

    def forward(self, img, init_mask, iters=3):
        img_feats = self.img_stem(img)
        curr_logits = (init_mask - 0.5) * 2.0 
        for _ in range(iters):
            curr_mask = torch.sigmoid(curr_logits)
            mask_feats = self.mask_stem(curr_mask)
            x = torch.cat([img_feats, mask_feats], dim=1)
            x = self.pre_loop(x)
            delta = self.looper(x)
            curr_logits = curr_logits + delta
        return torch.sigmoid(curr_logits)

class BoundarySmoother(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)
        logits = self.net(x)
        return torch.sigmoid(logits)

# ==========================================
# 2. LOAD MODELS & DATA
# ==========================================
print("\nLoading Models...")
fastsam_model = FastSAM("FastSAM-x.pt")

model_dir = r"D:\sems\AIP\Proj\code_nn\RCNN\kaggle codes\iter7"

model1 = BoundaryPuller().to(device)
model1.load_state_dict(torch.load(os.path.join(model_dir, "puller_final.pth"), map_location=device))
model1.eval()

model2 = BoundarySmoother().to(device)
model2.load_state_dict(torch.load(os.path.join(model_dir, "smoother_final.pth"), map_location=device))
model2.eval()

img_path = r"D:\sems\AIP\Proj\acceleration\i1.jpg"
mask_path = r"D:\sems\AIP\Proj\acceleration\m1.png"

img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W, _ = img_rgb.shape

fs_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
fs_bin = (fs_mask > 127).astype(np.float32).squeeze()

# ==========================================
# 3. FASTSAM BASELINE TIMING (Averaged)
# ==========================================
print("\n--- Running FastSAM Baseline (10 runs for average) ---")
_ = fastsam_model(img_path, device=device.type, verbose=False)
torch.cuda.synchronize()

t0 = time.time()
with torch.no_grad():
    for _ in range(10):
        _ = fastsam_model(img_path, device=device.type, verbose=False)
torch.cuda.synchronize()
fastsam_time = (time.time() - t0) / 10
print(f"FastSAM Average Inference Time: {fastsam_time:.4f} seconds")

# ==========================================
# 4. PREPARE PATCH COORDINATES
# ==========================================
fs_uint8 = (fs_bin * 255).astype(np.uint8)
contours, _ = cv2.findContours(fs_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

patch_size = 256
half = patch_size // 2
stride = 150 

patch_centers = []
for cnt in contours:
    for i in range(0, len(cnt), stride):
        patch_centers.append((cnt[i][0][0], cnt[i][0][1]))

print(f"\nTotal boundary patches to process per image: {len(patch_centers)}")

coords = []
img_crops = []
mask_crops = []

for x, y in patch_centers:
    x1, x2 = max(0, x - half), min(W, x + half)
    y1, y2 = max(0, y - half), min(H, y + half)
    
    if x2 - x1 < patch_size:
        if x1 == 0: x2 = min(W, patch_size)
        else: x1 = max(0, W - patch_size)
    if y2 - y1 < patch_size:
        if y1 == 0: y2 = min(H, patch_size)
        else: y1 = max(0, H - patch_size)
        
    coords.append((x1, y1, x2, y2))
    img_crops.append(img_rgb[y1:y2, x1:x2])
    mask_crops.append(fs_bin[y1:y2, x1:x2])

OPTIMAL_ITERS = 5
NUM_RUNS = 500

# ==========================================
# 5. UNOPTIMIZED TIMING (Batch Size = 1)
# ==========================================
print(f"\n--- Running Unoptimized Pipeline (BS=1) for {NUM_RUNS} iterations ---")
full_prob_unopt = fs_bin.copy()
full_count_unopt = np.ones((H, W), dtype=np.float32)

# GPU Warmup
with torch.no_grad():
    for _ in range(5):
        for i in range(len(coords)):
            x1, y1, x2, y2 = coords[i]
            img_t = torch.from_numpy(img_crops[i].astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask_crops[i]).view(1, 1, patch_size, patch_size).to(device)
            m1_out = model1(img_t, mask_t, iters=OPTIMAL_ITERS)
            m2_out = model2(img_t, (m1_out > 0.5).float())
            _ = m2_out.cpu()

torch.cuda.synchronize()
t0 = time.time()

with torch.no_grad():
    for _ in tqdm(range(NUM_RUNS), desc="Unopt Bench"):
        for i in range(len(coords)):
            x1, y1, x2, y2 = coords[i]
            
            img_t = torch.from_numpy(img_crops[i].astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask_crops[i]).view(1, 1, patch_size, patch_size).to(device)
            
            m1_out = model1(img_t, mask_t, iters=OPTIMAL_ITERS)
            m1_bin = (m1_out > 0.5).float()
            m2_out = model2(img_t, m1_bin)
            
            pred_patch = m2_out.squeeze().cpu().numpy()
            
            full_prob_unopt[y1:y2, x1:x2] += pred_patch
            full_count_unopt[y1:y2, x1:x2] += 1

torch.cuda.synchronize()
unopt_total_time = time.time() - t0
unopt_avg_time = unopt_total_time / NUM_RUNS
print(f"Average Unoptimized Time per image: {unopt_avg_time:.4f} seconds")

# ==========================================
# 6. OPTIMIZED TIMING (Batched Tensors)
# ==========================================
print(f"\n--- Running Optimized Pipeline (Batched) for {NUM_RUNS} iterations ---")
full_prob_opt = fs_bin.copy()
full_count_opt = np.ones((H, W), dtype=np.float32)
BATCH_SIZE = 32

all_imgs_t = torch.from_numpy(np.stack(img_crops).astype(np.float32) / 255.0).permute(0, 3, 1, 2)
all_masks_t = torch.from_numpy(np.stack(mask_crops)).view(-1, 1, patch_size, patch_size)

# GPU Warmup
with torch.no_grad():
    for _ in range(5):
        for i in range(0, len(coords), BATCH_SIZE):
            batch_img = all_imgs_t[i:i+BATCH_SIZE].to(device)
            batch_mask = all_masks_t[i:i+BATCH_SIZE].to(device)
            m1_out = model1(batch_img, batch_mask, iters=OPTIMAL_ITERS)
            m2_out = model2(batch_img, (m1_out > 0.5).float())
            _ = m2_out.cpu()

torch.cuda.synchronize()
t0 = time.time()

with torch.no_grad():
    for _ in tqdm(range(NUM_RUNS), desc="Opt Bench"):
        for i in range(0, len(coords), BATCH_SIZE):
            batch_img = all_imgs_t[i:i+BATCH_SIZE].to(device)
            batch_mask = all_masks_t[i:i+BATCH_SIZE].to(device)
            
            m1_out = model1(batch_img, batch_mask, iters=OPTIMAL_ITERS)
            m1_bin = (m1_out > 0.5).float()
            m2_out = model2(batch_img, m1_bin)
            
            pred_patches = m2_out.squeeze(1).cpu().numpy()
            
            for j in range(len(pred_patches)):
                x1, y1, x2, y2 = coords[i+j]
                full_prob_opt[y1:y2, x1:x2] += pred_patches[j]
                full_count_opt[y1:y2, x1:x2] += 1

torch.cuda.synchronize()
opt_total_time = time.time() - t0
opt_avg_time = opt_total_time / NUM_RUNS
print(f"Average Batched Optimized Time per image: {opt_avg_time:.4f} seconds")
print(f"True Speedup vs Unoptimized: {unopt_avg_time / opt_avg_time:.2f}x")

# ==========================================
# 7. FINAL MERGE & PLOTTING
# ==========================================
final_mask = ((full_prob_opt / full_count_opt) > 0.5).astype(np.uint8)

plt.figure(figsize=(15, 5), dpi=300)
plt.subplot(1, 3, 1)
plt.title(f"FastSAM Baseline\nTime: {fastsam_time:.3f}s")
plt.imshow(img_rgb)
plt.imshow(fs_bin, alpha=0.5, cmap='jet')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title(f"Refined Final Mask\nTime: {opt_avg_time:.3f}s (Batched)")
plt.imshow(img_rgb)
plt.imshow(final_mask, alpha=0.5, cmap='jet')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Overlap Heatmap")
plt.imshow(full_count_opt, cmap='hot')
plt.axis('off')

plt.tight_layout()
plt.show()