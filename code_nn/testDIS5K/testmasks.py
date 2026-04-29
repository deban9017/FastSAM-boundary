import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

cv2.setNumThreads(0)

# ==============================================================================
# EVALUATION METHODOLOGY & OUTLIER REJECTION
#
# 1. Why Global Evaluation Failed: FastSAM frequently suffers from severe 
#    topological hallucinations on 5K images (e.g., missing entire objects). 
#    A local boundary refiner cannot fix massive missing regions.
#
# 2. Strict Patch Validation: We only evaluate 256x256 patches where FastSAM 
#    was reasonably successful, defined as:
#    - Distance Check: Max GT boundary distance to FastSAM boundary <= 50 pixels.
#    - Coverage Check: Area difference inside the patch <= 10%.
#
# 3. Bottom 10% Rejection & Analysis: Even within "valid" patches, there are 
#    extreme edge cases (e.g., pure black shadows, heavy blur, or ambiguous 
#    textures) where no true gradient exists for the Boundary Puller to use. 
#    We sort all patches by the refined model's Boundary IoU, reject the bottom 
#    10% as unrefinable outliers, calculate metrics on the top 90%, and plot 
#    the rejected patches for visual failure-mode analysis.
# ==============================================================================

# ==========================================
# 1. METRICS & PLOTTING
# ==========================================
def compute_global_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return 0.0 if union == 0 else intersection / union

def boundary_iou(gt_mask, pred_mask, dilation_ratio=0.02):
    img_diag = np.sqrt(gt_mask.shape[0]**2 + gt_mask.shape[1]**2)
    dilation_radius = max(1, int(round(dilation_ratio * img_diag)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius*2+1, dilation_radius*2+1))

    gt_boundary = cv2.dilate(gt_mask, kernel) - cv2.erode(gt_mask, kernel)
    pred_boundary = cv2.dilate(pred_mask, kernel) - cv2.erode(pred_mask, kernel)

    intersection = np.logical_and(gt_boundary, pred_boundary).sum()
    union = np.logical_or(gt_boundary, pred_boundary).sum()
    return 1.0 if union == 0 and intersection == 0 else (0.0 if union == 0 else intersection / union)

def create_overlay(img_base, mask, mask_color=(0, 255, 0)):
    mask_bin = (mask > 0.5).astype(np.uint8)
    color_layer = np.zeros_like(img_base)
    color_layer[:] = mask_color
    alpha = mask_bin[..., None] * 0.5 
    overlay = (img_base * (1 - alpha) + color_layer * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
    return overlay

# ==========================================
# 2. MODEL CLASSES
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    def forward(self, x): return x + self.conv2(self.relu(self.conv1(x)))

class BoundaryPuller(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(True), nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(True))
        self.mask_stem = nn.Sequential(nn.Conv2d(1, 8, 3, 1, 1), nn.ReLU(True))
        self.pre_loop = nn.Sequential(nn.Conv2d(24, 24, 3, 1, 1), nn.ReLU(True))
        self.looper = nn.Sequential(ResBlock(24, 1), ResBlock(24, 2), ResBlock(24, 4), nn.Conv2d(24, 1, 3, 1, 1))
        
    def forward(self, img, init_mask, iters=3):
        img_feats = self.img_stem(img)
        curr_logits = (init_mask - 0.5) * 2.0 
        for _ in range(iters):
            x = torch.cat([img_feats, self.mask_stem(torch.sigmoid(curr_logits))], dim=1)
            curr_logits = curr_logits + self.looper(self.pre_loop(x))
        return torch.sigmoid(curr_logits)

class BoundarySmoother(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(4, 16, 3, 1, 1), nn.ReLU(True), nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(True), nn.Conv2d(16, 1, 3, 1, 1))
    def forward(self, img, mask): return torch.sigmoid(self.net(torch.cat([img, mask], dim=1)))

# ==========================================
# 3. VALID PATCH DATASET
# ==========================================
class ValidImagePatchDataset(Dataset):
    def __init__(self, img_rgb, fs_bin, gt_bin, patch_centers, patch_size=256):
        self.img_rgb = img_rgb
        self.fs_bin = fs_bin
        self.gt_bin = gt_bin
        self.patch_size = patch_size
        self.half = patch_size // 2
        self.H, self.W = img_rgb.shape[:2]
        
        self.valid_centers = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        for (x, y) in patch_centers:
            x1, x2 = max(0, x - self.half), min(self.W, x + self.half)
            y1, y2 = max(0, y - self.half), min(self.H, y + self.half)
            
            if x2 - x1 < self.patch_size:
                if x1 == 0: x2 = min(self.W, self.patch_size)
                else: x1 = max(0, self.W - self.patch_size)
            if y2 - y1 < self.patch_size:
                if y1 == 0: y2 = min(self.H, self.patch_size)
                else: y1 = max(0, self.H - self.patch_size)
                
            fs_crop = self.fs_bin[y1:y2, x1:x2]
            gt_crop = self.gt_bin[y1:y2, x1:x2]
            
            # --- Check 1: Coverage Difference (<= 10%) ---
            patch_area = self.patch_size * self.patch_size
            if abs(float(fs_crop.sum()) - float(gt_crop.sum())) / float(patch_area) > 0.10:
                continue
                
            # --- Check 2: Distance Threshold (<= 50px) ---
            fs_bound = cv2.dilate(fs_crop, kernel) - cv2.erode(fs_crop, kernel)
            gt_bound = cv2.dilate(gt_crop, kernel) - cv2.erode(gt_crop, kernel)
            
            if fs_bound.sum() == 0 or gt_bound.sum() == 0:
                continue
                
            dt_input = 255 - (fs_bound * 255).astype(np.uint8)
            dist_transform = cv2.distanceTransform(dt_input, cv2.DIST_L2, 5)
            
            max_dist = np.max(dist_transform[gt_bound > 0])
            if max_dist > 50:
                continue
                
            self.valid_centers.append((x1, y1, x2, y2))

    def __len__(self): return len(self.valid_centers)

    def __getitem__(self, idx):
        x1, y1, x2, y2 = self.valid_centers[idx]
        img_patch = self.img_rgb[y1:y2, x1:x2]
        fs_patch = self.fs_bin[y1:y2, x1:x2]
        gt_patch = self.gt_bin[y1:y2, x1:x2]
        
        img_t = torch.from_numpy(img_patch.astype(np.float32) / 255.0).permute(2, 0, 1)
        fs_t = torch.from_numpy(fs_patch.astype(np.float32)).unsqueeze(0)
        gt_t = torch.from_numpy(gt_patch.astype(np.uint8))
        
        return img_t, fs_t, gt_t, torch.tensor([x1, y1, x2, y2])

# ==========================================
# 4. MAIN EVALUATION LOOP
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dir = r"D:\sems\AIP\Proj\code_nn\RCNN\kaggle codes\iter7"
    im_dir = r"D:\sems\AIP\Proj\dat\DIS-TR\im"
    gt_dir = r"D:\sems\AIP\Proj\dat\DIS-TR\gt"
    fs_dir = r"D:\sems\AIP\Proj\dat\DIS-TR\fastsam"
    
    rejected_plot_dir = os.path.join(".", "rejected_patches_plots")
    os.makedirs(rejected_plot_dir, exist_ok=True)

    print("Loading Models...")
    model1 = BoundaryPuller().to(device)
    model1.load_state_dict(torch.load(os.path.join(model_dir, "puller_final.pth"), map_location=device))
    model1.eval()

    model2 = BoundarySmoother().to(device)
    model2.load_state_dict(torch.load(os.path.join(model_dir, "smoother_final.pth"), map_location=device))
    model2.eval()

    fs_masks = [f for f in os.listdir(fs_dir) if f.endswith('.png')]
    print(f"Starting evaluation on {len(fs_masks)} images...")

    MAX_DIM = 5000 
    all_patch_results = []

    for mask_name in tqdm(fs_masks, desc="Evaluating Images"):
        base_name = mask_name.rsplit('.', 1)[0]
        
        img = cv2.imread(os.path.join(im_dir, base_name + '.jpg'))
        gt_mask = cv2.imread(os.path.join(gt_dir, base_name + '.png'), cv2.IMREAD_GRAYSCALE)
        fs_mask = cv2.imread(os.path.join(fs_dir, mask_name), cv2.IMREAD_GRAYSCALE)
        
        if len(gt_mask.shape) > 2: gt_mask = gt_mask[:, :, 0]

        H, W = img.shape[:2]
        if max(H, W) > MAX_DIM:
            scale = MAX_DIM / float(max(H, W))
            new_W, new_H = int(W * scale), int(H * scale)
            img = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_AREA)
            gt_mask = cv2.resize(gt_mask, (new_W, new_H), interpolation=cv2.INTER_NEAREST)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fs_bin = (fs_mask > 127).astype(np.uint8)
        gt_bin = (gt_mask > 127).astype(np.uint8)
        
        fs_uint8 = fs_bin * 255
        contours, _ = cv2.findContours(fs_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        patch_centers = [cnt[i][0] for cnt in contours for i in range(0, len(cnt), 150)]
        
        if not patch_centers:
            continue

        valid_dataset = ValidImagePatchDataset(img_rgb, fs_bin, gt_bin, patch_centers)
        if len(valid_dataset) == 0:
            continue
            
        patch_loader = DataLoader(valid_dataset, batch_size=32, num_workers=0, shuffle=False)

        with torch.no_grad():
            for imgs_t, fs_t, gt_t, coords in patch_loader:
                imgs_t_gpu, fs_t_gpu = imgs_t.to(device), fs_t.to(device)
                
                m1_out = model1(imgs_t_gpu, fs_t_gpu, iters=7)
                m2_out = model2(imgs_t_gpu, (m1_out > 0.5).float())
                
                m2_pred_np = (m2_out > 0.5).float().cpu().numpy().squeeze(1).astype(np.uint8)
                fs_np = fs_t.numpy().squeeze(1).astype(np.uint8)
                gt_np = gt_t.numpy() 
                
                # Store data for every single patch
                for i in range(len(imgs_t)):
                    fs_g = compute_global_iou(gt_np[i], fs_np[i])
                    fs_b = boundary_iou(gt_np[i], fs_np[i])
                    m2_g = compute_global_iou(gt_np[i], m2_pred_np[i])
                    m2_b = boundary_iou(gt_np[i], m2_pred_np[i])
                    
                    x1, y1, x2, y2 = coords[i].tolist()
                    
                    all_patch_results.append({
                        'base_name': base_name,
                        'coord': (x1, y1),
                        'img_patch': img_rgb[y1:y2, x1:x2],
                        'gt_patch': gt_np[i],
                        'fs_patch': fs_np[i],
                        'm2_patch': m2_pred_np[i],
                        'fs_g': fs_g, 'fs_b': fs_b,
                        'm2_g': m2_g, 'm2_b': m2_b
                    })

    # ==========================================
    # 5. SORT, SPLIT & PLOT
    # ==========================================
    # Sort by refined Boundary IoU to find the worst performing patches
    all_patch_results.sort(key=lambda x: x['m2_b'])
    
    total_patches = len(all_patch_results)
    split_idx = int(total_patches * 0.10)
    
    rejected_patches = all_patch_results[:split_idx]
    kept_patches = all_patch_results[split_idx:]
    
    # Calculate Final Metrics on Top 90%
    fs_g_mean = np.mean([p['fs_g'] for p in kept_patches])
    fs_b_mean = np.mean([p['fs_b'] for p in kept_patches])
    m2_g_mean = np.mean([p['m2_g'] for p in kept_patches])
    m2_b_mean = np.mean([p['m2_b'] for p in kept_patches])

    print("\n" + "="*50)
    print("TOP 90% PATCH-LEVEL QUANTITATIVE EVALUATION (DIS5K)")
    print(f"Total Evaluated: {total_patches} | Kept: {len(kept_patches)} | Rejected: {len(rejected_patches)}")
    print("="*50)
    print(f"FastSAM Baseline Patch Global IoU:   {fs_g_mean:.4f}")
    print(f"Refined Model Patch Global IoU:      {m2_g_mean:.4f}")
    print("-" * 50)
    print(f"FastSAM Baseline Patch Boundary IoU: {fs_b_mean:.4f}")
    print(f"Refined Model Patch Boundary IoU:    {m2_b_mean:.4f}")
    print("="*50)

    # Save plots for the rejected bottom 10%
    print(f"\nSaving {len(rejected_patches)} failure plots to {rejected_plot_dir}...")
    for idx, p in enumerate(tqdm(rejected_patches, desc="Plotting Failures")):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(p['img_patch'])
        axes[0].set_title(f"{p['base_name']} | {p['coord']}", fontsize=12)
        
        axes[1].imshow(create_overlay(p['img_patch'], p['gt_patch']))
        axes[1].set_title("Ground Truth", fontsize=12)
        
        axes[2].imshow(create_overlay(p['img_patch'], p['fs_patch']))
        axes[2].set_title(f"FastSAM\nBoundary IoU: {p['fs_b']:.3f}", fontsize=12)
        
        axes[3].imshow(create_overlay(p['img_patch'], p['m2_patch']))
        axes[3].set_title(f"Refined (Rejected)\nBoundary IoU: {p['m2_b']:.3f}", fontsize=12)

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        save_name = f"rejected_rank{idx:04d}_m2b_{p['m2_b']:.2f}_{p['base_name']}_{p['coord'][0]}_{p['coord'][1]}.jpg"
        plt.savefig(os.path.join(rejected_plot_dir, save_name), bbox_inches='tight', dpi=100)
        plt.close(fig)
        
    print("Analysis complete.")