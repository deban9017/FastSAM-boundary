import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==========================================
# 1. MODEL CLASSES (Iter 7 Standard)
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
# 2. UTILS & SETUP
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_dir = r"D:\sems\AIP\Proj\code_nn\RCNN\parallelCorrector"

# ==========================================
# 3. LOAD MODELS
# ==========================================
print("Loading models...")
model1 = BoundaryPuller().to(device)
model1.load_state_dict(torch.load(os.path.join(base_dir, "iter2models/puller_final.pth"), map_location=device))
model1.eval()

model2 = BoundarySmoother().to(device)
model2.load_state_dict(torch.load(os.path.join(base_dir, "iter2models/smoother_final.pth"), map_location=device))
model2.eval()

# ==========================================
# 4. LOAD IMAGES
# ==========================================
img_path = os.path.join(base_dir, "sa_2001_a3.jpg")
mask_path = os.path.join(base_dir, "sa_2001_a3.png")

img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fs_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
fs_bin = (fs_mask > 127).astype(np.uint8)

# Convert to Tensors
img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
fs_t = torch.from_numpy(fs_bin.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

# ==========================================
# 5. INFERENCE
# ==========================================
print("Running inference...")
with torch.no_grad():
    # Run Puller
    m1_pred_t = model1(img_t, fs_t, iters=100)
    m1_bin_t = (m1_pred_t > 0.5).float()
    
    # Run Smoother
    m2_pred_t = model2(img_t, m1_bin_t)
    
    m1_out = m1_pred_t.squeeze().cpu().numpy()
    m2_out = m2_pred_t.squeeze().cpu().numpy()

m1_bin_np = (m1_out > 0.5).astype(np.uint8)
m2_bin_np = (m2_out > 0.5).astype(np.uint8)

# ==========================================
# 6. VISUALIZATION & SAVE
# ==========================================
print("Generating and saving plot...")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(img_rgb)
axes[0].set_title("Input")

axes[1].imshow(create_overlay(img_rgb, fs_bin))
axes[1].set_title("FastSAM")

axes[2].imshow(create_overlay(img_rgb, m1_bin_np))
axes[2].set_title("M1: Pulled")

axes[3].imshow(create_overlay(img_rgb, m2_bin_np))
axes[3].set_title("M2: Smoothed")

for ax in axes:
    ax.axis('off')

plt.tight_layout()
save_path = os.path.join(base_dir, "test_result_parallel.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved successfully to: {save_path}")
plt.show()