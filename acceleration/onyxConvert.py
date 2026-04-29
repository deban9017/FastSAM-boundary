import torch
import torch.nn as nn
import os

# ==========================================
# MODEL CLASSES (Same as before)
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
    def forward(self, img, init_mask, iters=5): # Hardcoded default to 5 for ONNX unrolling
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
# EXPORT LOGIC
# ==========================================
print("Loading PyTorch Models...")
model_dir = r"D:\sems\AIP\Proj\code_nn\RCNN\kaggle codes\iter7"

puller = BoundaryPuller().eval()
puller.load_state_dict(torch.load(os.path.join(model_dir, "puller_final.pth"), map_location='cpu'))

smoother = BoundarySmoother().eval()
smoother.load_state_dict(torch.load(os.path.join(model_dir, "smoother_final.pth"), map_location='cpu'))

# Create dummy inputs that match the size of your batches
# Batch size 32, Channels 3/1, 256x256 patches
dummy_img = torch.randn(32, 3, 256, 256)
dummy_mask = torch.randn(32, 1, 256, 256)

print("Exporting Puller to ONNX (unrolling 5 iterations)...")
torch.onnx.export(
    puller, 
    (dummy_img, dummy_mask), 
    "puller_opt.onnx",
    input_names=['image', 'mask'],
    output_names=['output'],
    opset_version=11,
    dynamic_axes={'image': {0: 'batch_size'}, 'mask': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("Exporting Smoother to ONNX...")
torch.onnx.export(
    smoother, 
    (dummy_img, dummy_mask), 
    "smoother_opt.onnx",
    input_names=['image', 'mask'],
    output_names=['output'],
    opset_version=11,
    dynamic_axes={'image': {0: 'batch_size'}, 'mask': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# Export FastSAM
from ultralytics import FastSAM
print("Exporting FastSAM to ONNX...")
fs_model = FastSAM("FastSAM-x.pt")
fs_model.export(format="onnx") # Ultralytics handles its own complex export

print("\nSuccess! ONNX models generated.")