"""
OUTPUT: (Timing on CPU)

Total boundary patches to process: 15

--- Running CPU Pipeline (Batched) for 50 iterations ---
CPU Bench: 100%|███████████████████████████████████████████████████████████████████| 50/50 [01:30<00:00,  1.82s/it]

======================================
Average CPU Inference Time: 1.8164 seconds
======================================

"""

# ===== CLEAR MEMORY =====
import gc
import os
import time
import torch
import torch.nn as nn
import cv2
import numpy as np
from tqdm import tqdm

gc.collect()

# FORCING CPU
device = torch.device('cpu')
print("Running purely on:", device)

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
        return torch.sigmoid(self.net(x))

# ==========================================
# 2. LOAD MODELS & DATA
# ==========================================
print("\nLoading Models...")
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
# 3. PREPARE PATCH COORDINATES
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

print(f"\nTotal boundary patches to process: {len(patch_centers)}")

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
NUM_RUNS = 50 # Dropped to 200 so we don't wait forever if CPU is slow

# ==========================================
# 4. OPTIMIZED TIMING ON CPU
# ==========================================
print(f"\n--- Running CPU Pipeline (Batched) for {NUM_RUNS} iterations ---")

# Pre-stack all numpy arrays into a single massive tensor block
all_imgs_t = torch.from_numpy(np.stack(img_crops).astype(np.float32) / 255.0).permute(0, 3, 1, 2)
all_masks_t = torch.from_numpy(np.stack(mask_crops)).view(-1, 1, patch_size, patch_size)

BATCH_SIZE = 32

t0 = time.time()

with torch.no_grad():
    for _ in tqdm(range(NUM_RUNS), desc="CPU Bench"):
        full_prob_opt = fs_bin.copy()
        full_count_opt = np.ones((H, W), dtype=np.float32)
        
        for i in range(0, len(coords), BATCH_SIZE):
            batch_img = all_imgs_t[i:i+BATCH_SIZE]
            batch_mask = all_masks_t[i:i+BATCH_SIZE]
            
            m1_out = model1(batch_img, batch_mask, iters=OPTIMAL_ITERS)
            m1_bin = (m1_out > 0.5).float()
            m2_out = model2(batch_img, m1_bin)
            
            # Already on CPU, just convert to numpy
            pred_patches = m2_out.squeeze(1).numpy()
            
            for j in range(len(pred_patches)):
                x1, y1, x2, y2 = coords[i+j]
                full_prob_opt[y1:y2, x1:x2] += pred_patches[j]
                full_count_opt[y1:y2, x1:x2] += 1

cpu_total_time = time.time() - t0
cpu_avg_time = cpu_total_time / NUM_RUNS

print(f"\n======================================")
print(f"Average CPU Inference Time: {cpu_avg_time:.4f} seconds")
print(f"======================================")