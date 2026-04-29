import os
import gc
import cv2
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 1. METRIC & PLOTTING FUNCTIONS
# ==========================================
def compute_global_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return 0.0 if union == 0 else intersection / union

def boundary_iou(gt_mask, pred_mask, dilation_ratio=0.02):
    img_diag = np.sqrt(gt_mask.shape[0]**2 + gt_mask.shape[1]**2)
    dilation_radius = int(round(dilation_ratio * img_diag))
    if dilation_radius < 1:
        dilation_radius = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius*2+1, dilation_radius*2+1))

    gt_boundary = cv2.dilate(gt_mask, kernel) - cv2.erode(gt_mask, kernel)
    pred_boundary = cv2.dilate(pred_mask, kernel) - cv2.erode(pred_mask, kernel)

    intersection = np.logical_and(gt_boundary, pred_boundary).sum()
    union = np.logical_or(gt_boundary, pred_boundary).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

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
# 3. SETUP & EVALUATION LOOP
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
model_dir = r"D:\sems\AIP\Proj\code_nn\RCNN\kaggle codes\iter7"
hqseg_dir = r"D:\sems\AIP\Proj\dat\HQSeg"
fsHQSeg_dir = r"D:\sems\AIP\Proj\dat\fsHQSeg"

# Directory for plots in current directory
plot_dir = os.path.join(".", "eval_plots")
os.makedirs(plot_dir, exist_ok=True)

# Load Models
print("Loading Refinement Models...")
model1 = BoundaryPuller().to(device)
model1.load_state_dict(torch.load(os.path.join(model_dir, "puller_final.pth"), map_location=device))
model1.eval()

model2 = BoundarySmoother().to(device)
model2.load_state_dict(torch.load(os.path.join(model_dir, "smoother_final.pth"), map_location=device))
model2.eval()

fs_masks = [f for f in os.listdir(fsHQSeg_dir) if f.endswith('.png')]
print(f"Starting evaluation on {len(fs_masks)} pairs...")

# Pick 10 random masks to plot
sample_masks_to_plot = set(random.sample(fs_masks, min(10, len(fs_masks))))

# Metric Trackers
metrics = {
    'fs_global': [], 'fs_boundary': [],
    'm2_global': [], 'm2_boundary': []
}

with torch.no_grad():
    for mask_name in tqdm(fs_masks, desc="Evaluating"):
        base_name = mask_name.rsplit('.', 1)[0]
        img_path = os.path.join(hqseg_dir, base_name + '.jpg')
        gt_path = os.path.join(hqseg_dir, mask_name)
        fs_path = os.path.join(fsHQSeg_dir, mask_name)
        
        # Load data
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fs_mask = cv2.imread(fs_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        # Force 2D for GT
        if len(gt_mask.shape) > 2:
            gt_mask = gt_mask[:, :, 0]
            
        fs_bin = (fs_mask > 127).astype(np.uint8)
        gt_bin = (gt_mask > 127).astype(np.uint8)
        
        # Calculate baseline FastSAM metrics
        metrics['fs_global'].append(compute_global_iou(gt_bin, fs_bin))
        metrics['fs_boundary'].append(boundary_iou(gt_bin, fs_bin))
        
        # Tensor prep
        img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        fs_t = torch.from_numpy(fs_bin.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        
        # Forward pass
        m1_pred = model1(img_t, fs_t, iters=7)
        m1_bin_t = (m1_pred > 0.5).float()
        m2_pred = model2(img_t, m1_bin_t)
        
        m2_out = m2_pred.squeeze().cpu().numpy()
        m2_bin_np = (m2_out > 0.5).astype(np.uint8)
        
        # Calculate Refined metrics
        metrics['m2_global'].append(compute_global_iou(gt_bin, m2_bin_np))
        metrics['m2_boundary'].append(boundary_iou(gt_bin, m2_bin_np))

        # --- Plotting Logic ---
        if mask_name in sample_masks_to_plot:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(img_rgb)
            axes[0].set_title("Input Image")
            
            axes[1].imshow(create_overlay(img_rgb, gt_bin))
            axes[1].set_title("Ground Truth")
            
            axes[2].imshow(create_overlay(img_rgb, fs_bin))
            axes[2].set_title("FastSAM")
            
            axes[3].imshow(create_overlay(img_rgb, m2_bin_np))
            axes[3].set_title("Our Model (Refined)")

            for ax in axes:
                ax.axis('off')

            plt.tight_layout()
            save_path = os.path.join(plot_dir, f"plot_{base_name}.jpg")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig) # Prevent memory leaks

# Print Final Report
print("\n" + "="*50)
print("QUANTITATIVE EVALUATION RESULTS")
print("="*50)
print(f"FastSAM Baseline Global IoU:   {np.mean(metrics['fs_global']):.4f}")
print(f"Refined Model Global IoU:      {np.mean(metrics['m2_global']):.4f}")
print("-" * 50)
print(f"FastSAM Baseline Boundary IoU: {np.mean(metrics['fs_boundary']):.4f}")
print(f"Refined Model Boundary IoU:    {np.mean(metrics['m2_boundary']):.4f}")
print("="*50)