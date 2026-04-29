print("Starting script...")
import os
import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

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
# 3. DATASET
# ==========================================
class ImagePatchDataset(Dataset):
    def __init__(self, img_rgb, fs_bin, patch_centers, patch_size=256):
        self.img_rgb = img_rgb
        self.fs_bin = fs_bin
        self.patch_centers = patch_centers
        self.patch_size = patch_size
        self.half = patch_size // 2
        self.H, self.W = img_rgb.shape[:2]

    def __len__(self):
        return len(self.patch_centers)

    def __getitem__(self, idx):
        x, y = self.patch_centers[idx]
        x1, x2 = max(0, x - self.half), min(self.W, x + self.half)
        y1, y2 = max(0, y - self.half), min(self.H, y + self.half)
        
        if x2 - x1 < self.patch_size:
            if x1 == 0: x2 = min(self.W, self.patch_size)
            else: x1 = max(0, self.W - self.patch_size)
        if y2 - y1 < self.patch_size:
            if y1 == 0: y2 = min(self.H, self.patch_size)
            else: y1 = max(0, self.H - self.patch_size)
            
        img_patch = self.img_rgb[y1:y2, x1:x2]
        mask_patch = self.fs_bin[y1:y2, x1:x2]
        
        img_t = torch.from_numpy(img_patch.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask_patch.astype(np.float32)).unsqueeze(0)
        
        return img_t, mask_t, torch.tensor([x1, y1, x2, y2])

# ==========================================
# 4. MAIN LOOP
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dir = r"D:\sems\AIP\Proj\code_nn\RCNN\kaggle codes\iter7"
    im_dir = r"D:\sems\AIP\Proj\dat\DIS-TR\im"
    gt_dir = r"D:\sems\AIP\Proj\dat\DIS-TR\gt"
    fs_dir = r"D:\sems\AIP\Proj\dat\DIS-TR\fastsam"
    
    # Create plot directory
    plot_dir = os.path.join(".", "dis5k_plots")
    os.makedirs(plot_dir, exist_ok=True)

    print("Loading Models...")
    model1 = BoundaryPuller().to(device)
    model1.load_state_dict(torch.load(os.path.join(model_dir, "puller_final.pth"), map_location=device))
    model1.eval()

    model2 = BoundarySmoother().to(device)
    model2.load_state_dict(torch.load(os.path.join(model_dir, "smoother_final.pth"), map_location=device))
    model2.eval()

    fs_masks = [f for f in os.listdir(fs_dir) if f.endswith('.png')]
    print(f"Starting evaluation on {len(fs_masks)} images...")

    metrics = {'fs_global': [], 'fs_boundary': [], 'm2_global': [], 'm2_boundary': []}
    MAX_DIM = 5000 

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
        H_new, W_new = img_rgb.shape[:2]
            
        fs_bin = (fs_mask > 127).astype(np.float32)
        gt_bin = (gt_mask > 127).astype(np.uint8)
        
        # Calculate stats for the plot
        fs_area = fs_bin.sum()
        gt_area = gt_bin.sum()
        area_diff_ratio = abs(float(fs_area) - float(gt_area)) / max(float(gt_area), 1.0)
        
        fs_g_iou = compute_global_iou(gt_bin, fs_bin.astype(np.uint8))
        fs_b_iou = boundary_iou(gt_bin, fs_bin.astype(np.uint8))
        
        metrics['fs_global'].append(fs_g_iou)
        metrics['fs_boundary'].append(fs_b_iou)

        fs_uint8 = (fs_bin * 255).astype(np.uint8)
        contours, _ = cv2.findContours(fs_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        patch_centers = [cnt[i][0] for cnt in contours for i in range(0, len(cnt), 150)]
        
        if not patch_centers:
            continue

        # Zero workers to prevent Windows multiprocessing lockup inside the loop
        patch_dataset = ImagePatchDataset(img_rgb, fs_bin, patch_centers)
        patch_loader = DataLoader(patch_dataset, batch_size=32, num_workers=0, shuffle=False)
        
        full_prob = fs_bin.copy()
        full_count = np.ones((H_new, W_new), dtype=np.float32)

        with torch.no_grad():
            for imgs_t, masks_t, coords in patch_loader:
                imgs_t, masks_t = imgs_t.to(device), masks_t.to(device)
                
                m1_out = model1(imgs_t, masks_t, iters=7)
                m2_out = model2(imgs_t, (m1_out > 0.5).float())
                
                pred_patches = m2_out.squeeze(1).cpu().numpy()
                
                for i in range(len(coords)):
                    x1, y1, x2, y2 = coords[i].tolist()
                    full_prob[y1:y2, x1:x2] += pred_patches[i]
                    full_count[y1:y2, x1:x2] += 1
                    
        final_mask = ((full_prob / full_count) > 0.5).astype(np.uint8)
        
        m2_g_iou = compute_global_iou(gt_bin, final_mask)
        m2_b_iou = boundary_iou(gt_bin, final_mask)
        
        metrics['m2_global'].append(m2_g_iou)
        metrics['m2_boundary'].append(m2_b_iou)

        # --- Plotting & Saving ---
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        axes[0].imshow(img_rgb)
        axes[0].set_title("Input Image", fontsize=14)
        
        axes[1].imshow(create_overlay(img_rgb, gt_bin))
        axes[1].set_title(f"Ground Truth\nArea: {int(gt_area)} px", fontsize=14)
        
        axes[2].imshow(create_overlay(img_rgb, fs_bin))
        axes[2].set_title(f"FastSAM Baseline\nIoU: {fs_g_iou:.3f} | Area Diff: {area_diff_ratio:.1%}", fontsize=14)
        
        axes[3].imshow(create_overlay(img_rgb, final_mask))
        axes[3].set_title(f"Our Model (Refined)\nGlobal IoU: {m2_g_iou:.3f} | Boundary IoU: {m2_b_iou:.3f}", fontsize=14)

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        save_path = os.path.join(plot_dir, f"plot_{base_name}.jpg")
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close(fig) # Prevent memory leaks

    print("\n" + "="*50)
    print("QUANTITATIVE EVALUATION RESULTS (DIS5K)")
    print("="*50)
    print(f"FastSAM Baseline Global IoU:   {np.mean(metrics['fs_global']):.4f}")
    print(f"Refined Model Global IoU:      {np.mean(metrics['m2_global']):.4f}")
    print("-" * 50)
    print(f"FastSAM Baseline Boundary IoU: {np.mean(metrics['fs_boundary']):.4f}")
    print(f"Refined Model Boundary IoU:    {np.mean(metrics['m2_boundary']):.4f}")
    print("="*50)