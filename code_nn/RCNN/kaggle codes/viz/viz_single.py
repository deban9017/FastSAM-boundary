import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
ITERS_TO_RUN = 1000  # Change this to test different pulling iterations

IMG_PATH = r"D:\sems\AIP\Proj\code_nn\RCNN\kaggle codes\viz\sa_2331_a1.jpg"
MASK_PATH = r"D:\sems\AIP\Proj\code_nn\RCNN\kaggle codes\viz\sa_2331_a1.png"
MODEL_DIR = r"D:\sems\AIP\Proj\code_nn\RCNN\kaggle codes\iter7"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# MODEL CLASSES (Original Iter 7)
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
            ResBlock(24, dilation=1),
            ResBlock(24, dilation=2),
            ResBlock(24, dilation=4),
            nn.Conv2d(24, 1, kernel_size=3, padding=1)
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
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)
        logits = self.net(x)
        return torch.sigmoid(logits)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
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
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Loading Models...")
    model1 = BoundaryPuller().to(device)
    model1.load_state_dict(torch.load(os.path.join(MODEL_DIR, "puller_final.pth"), map_location=device))
    model1.eval()

    model2 = BoundarySmoother().to(device)
    model2.load_state_dict(torch.load(os.path.join(MODEL_DIR, "smoother_final.pth"), map_location=device))
    model2.eval()

    print(f"Loading Image: {IMG_PATH}")
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {IMG_PATH}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print(f"Loading FastSAM Mask: {MASK_PATH}")
    fs_mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    if fs_mask is None:
        raise FileNotFoundError(f"Could not load mask at {MASK_PATH}")
    fs_bin = (fs_mask > 127).astype(np.uint8)

    # Prepare Tensors
    img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    fs_t = torch.from_numpy(fs_bin.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    print(f"Running Inference (Puller Iters = {ITERS_TO_RUN})...")
    with torch.no_grad():
        # Stage 1: Puller
        m1_pred_t = model1(img_t, fs_t, iters=ITERS_TO_RUN)
        m1_bin_t = (m1_pred_t > 0.5).float()
        
        # Stage 2: Smoother
        m2_pred_t = model2(img_t, m1_bin_t)
        
        # Convert back to numpy
        final_mask_np = m2_pred_t.squeeze().cpu().numpy()
        final_bin = (final_mask_np > 0.5).astype(np.uint8)

    # Plotting
    print("Plotting results...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Image", fontsize=14)

    axes[1].imshow(create_overlay(img_rgb, fs_bin, mask_color=(255, 0, 0)))
    axes[1].set_title("FastSAM Mask", fontsize=14)

    axes[2].imshow(create_overlay(img_rgb, final_bin, mask_color=(0, 255, 0)))
    axes[2].set_title(f"Refined Mask (Puller x{ITERS_TO_RUN} + Smoother)", fontsize=14)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()