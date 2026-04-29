import torch
import torch.nn as nn
import os

# ==========================================
# MODEL CLASSES
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    def forward(self, x): return x + self.conv2(self.relu(self.conv1(x)))

class BoundaryPullerExplicit(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(True), nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(True))
        self.mask_stem = nn.Sequential(nn.Conv2d(1, 8, 3, 1, 1), nn.ReLU(True))
        self.pre_loop = nn.Sequential(nn.Conv2d(24, 24, 3, 1, 1), nn.ReLU(True))
        self.looper = nn.Sequential(ResBlock(24, 1), ResBlock(24, 2), ResBlock(24, 4), nn.Conv2d(24, 1, 3, 1, 1))

    def forward(self, img, init_mask):
        img_feats = self.img_stem(img)
        curr_logits = (init_mask - 0.5) * 2.0 
        
        # Iteration 1
        x = torch.cat([img_feats, self.mask_stem(torch.sigmoid(curr_logits))], dim=1)
        curr_logits = curr_logits + self.looper(self.pre_loop(x))
        # Iteration 2
        x = torch.cat([img_feats, self.mask_stem(torch.sigmoid(curr_logits))], dim=1)
        curr_logits = curr_logits + self.looper(self.pre_loop(x))
        # Iteration 3
        x = torch.cat([img_feats, self.mask_stem(torch.sigmoid(curr_logits))], dim=1)
        curr_logits = curr_logits + self.looper(self.pre_loop(x))
        # Iteration 4
        x = torch.cat([img_feats, self.mask_stem(torch.sigmoid(curr_logits))], dim=1)
        curr_logits = curr_logits + self.looper(self.pre_loop(x))
        # Iteration 5
        x = torch.cat([img_feats, self.mask_stem(torch.sigmoid(curr_logits))], dim=1)
        curr_logits = curr_logits + self.looper(self.pre_loop(x))

        return torch.sigmoid(curr_logits)

class BoundarySmoother(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(4, 16, 3, 1, 1), nn.ReLU(True), nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(True), nn.Conv2d(16, 1, 3, 1, 1))
    def forward(self, img, mask): 
        return torch.sigmoid(self.net(torch.cat([img, mask], dim=1)))

# === NEW: THE UNIFIED PIPELINE ===
class FullPipelineExplicit(nn.Module):
    def __init__(self):
        super().__init__()
        self.puller = BoundaryPullerExplicit()
        self.smoother = BoundarySmoother()

    def forward(self, img, init_mask):
        # 1. Run the fully unrolled Puller
        m1_out = self.puller(img, init_mask)
        
        # 2. Hard threshold binarization (ONNX supports this natively as a Cast/Greater node)
        m1_bin = (m1_out > 0.5).float()
        
        # 3. Run the Smoother
        m2_out = self.smoother(img, m1_bin)
        
        return m2_out

# ==========================================
# EXPORT LOGIC
# ==========================================
print("Loading PyTorch Models into Unified Pipeline...")
model_dir = r"D:\sems\AIP\Proj\code_nn\RCNN\kaggle codes\iter7"

# Instantiate the single master pipeline
pipeline = FullPipelineExplicit().eval()

# Load weights directly into the sub-modules
pipeline.puller.load_state_dict(torch.load(os.path.join(model_dir, "puller_final.pth"), map_location='cpu', weights_only=True))
pipeline.smoother.load_state_dict(torch.load(os.path.join(model_dir, "smoother_final.pth"), map_location='cpu', weights_only=True))

dummy_img = torch.randn(32, 3, 256, 256)
dummy_mask = torch.randn(32, 1, 256, 256)

print("Exporting Unified Pipeline to ONNX...")
torch.onnx.export(
    pipeline, 
    (dummy_img, dummy_mask), 
    "full_pipeline_optcat5.onnx",
    input_names=['image', 'mask'],
    output_names=['output'],
    opset_version=11,
    dynamic_axes={'image': {0: 'batch_size'}, 'mask': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("\nSuccess! Unified ONNX model generated.")