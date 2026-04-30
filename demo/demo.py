import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import gradio as gr
from ultralytics import YOLO 
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. MODEL CLASSES (Iter 7 Standard)
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
        self.img_stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(True), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(True))
        self.mask_stem = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(True))
        self.pre_loop = nn.Sequential(nn.Conv2d(24, 24, 3, padding=1), nn.ReLU(True))
        self.looper = nn.Sequential(ResBlock(24, 1), ResBlock(24, 2), ResBlock(24, 4), nn.Conv2d(24, 1, 3, padding=1))

    def forward(self, img, init_mask, iters=10):
        img_feats = self.img_stem(img)
        curr_logits = (init_mask - 0.5) * 2.0 
        for _ in range(iters):
            x = torch.cat([img_feats, self.mask_stem(torch.sigmoid(curr_logits))], dim=1)
            curr_logits = curr_logits + self.looper(self.pre_loop(x))
        return torch.sigmoid(curr_logits)

class BoundarySmoother(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(True), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(True), nn.Conv2d(16, 1, 3, padding=1))
    def forward(self, img, mask): return torch.sigmoid(self.net(torch.cat([img, mask], dim=1)))

# ==========================================
# 2. LOAD MODELS
# ==========================================
print("Loading FastSAM...")
fastsam_model = YOLO(r"D:\sems\AIP\Proj\FastSAM-x.pt")

print("Loading Refinement Models...")
puller = BoundaryPuller().to(device)
puller.load_state_dict(torch.load(r"D:\sems\AIP\Proj\demo\puller_final.pth", map_location=device))
puller.eval()

smoother = BoundarySmoother().to(device)
smoother.load_state_dict(torch.load(r"D:\sems\AIP\Proj\demo\smoother_final.pth", map_location=device))
smoother.eval()

# Global state for zooming
current_fs_overlay = None
current_refined_overlay = None

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def create_overlay(img_base, mask_bin):
    color_layer = np.zeros_like(img_base)
    color_layer[:] = (0, 255, 0)
    alpha = mask_bin[..., None] * 0.5 
    overlay = (img_base * (1 - alpha) + color_layer * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
    return overlay

def segment_on_click(img, evt: gr.SelectData):
    global current_fs_overlay, current_refined_overlay
    if img is None: return None, None, None
    
    click_x, click_y = evt.index
    current_img = img.copy()
    
    H, W, _ = current_img.shape
    
    # PROACTIVE OOM PREVENTION: Lowered threshold to 2500 for safety
    if H > 2500 or W > 2500:
        gr.Warning(f"Image is quite large ({W}x{H}). Automatically downsampling by 2x to prevent memory issues.")
        current_img = cv2.resize(current_img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        click_x = click_x // 2
        click_y = click_y // 2
        
    max_retries = 2
    for attempt in range(max_retries):
        try:
            H, W, _ = current_img.shape
            
            # 1. Run FastSAM
            # FIX: retina_masks=False to stop YOLO from allocating 3GB of RAM!
            results = fastsam_model(current_img, device=device, retina_masks=False, imgsz=1024, conf=0.4)
            if results[0].masks is None:
                gr.Warning("No object found at clicked location.")
                return current_img, current_img, current_img
                
            # 2. Find the mask that contains the clicked point
            masks_tensor = results[0].masks.data.cpu().numpy()
            target_mask = None
            
            for i in range(masks_tensor.shape[0]):
                m = cv2.resize(masks_tensor[i], (W, H), interpolation=cv2.INTER_NEAREST)
                if m[click_y, click_x] > 0.5:
                    target_mask = m
                    break
                    
            if target_mask is None:
                gr.Warning("No object found at clicked location.")
                return current_img, current_img, current_img

            fs_bin = (target_mask > 0.5).astype(np.uint8)
            
            # 3. Patch-based Refinement
            contours, _ = cv2.findContours(fs_bin * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            patch_centers = [(cnt[i][0][0], cnt[i][0][1]) for cnt in contours for i in range(0, len(cnt), 150)]
            
            img_t = torch.from_numpy(current_img.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
            fs_t = torch.from_numpy(fs_bin.astype(np.float32)).to(device)
            
            full_prob = fs_bin.copy().astype(np.float32)
            full_count = np.ones((H, W), dtype=np.float32)
            
            with torch.no_grad():
                for cx, cy in patch_centers:
                    x1, x2 = max(0, cx - 128), min(W, cx + 128)
                    y1, y2 = max(0, cy - 128), min(H, cy + 128)
                    if x2 - x1 < 256: x1, x2 = (0, 256) if x1 == 0 else (W - 256, W)
                    if y2 - y1 < 256: y1, y2 = (0, 256) if y1 == 0 else (H - 256, H)
                    
                    p_img = img_t[:, y1:y2, x1:x2].unsqueeze(0)
                    p_mask = fs_t[y1:y2, x1:x2].unsqueeze(0).unsqueeze(0)
                    
                    m1_pred = puller(p_img, p_mask, iters=10)
                    m2_pred = smoother(p_img, (m1_pred > 0.5).float())
                    
                    out_patch = m2_pred.squeeze().cpu().numpy()
                    full_prob[y1:y2, x1:x2] += out_patch
                    full_count[y1:y2, x1:x2] += 1
                    
            final_bin = ((full_prob / full_count) > 0.5).astype(np.uint8)
            
            # 4. Create Overlays
            current_fs_overlay = create_overlay(current_img, fs_bin)
            current_refined_overlay = create_overlay(current_img, final_bin)
            
            return current_fs_overlay, current_refined_overlay, current_fs_overlay
            
        except Exception as e:
            # FIX: Catch BOTH CPU and GPU memory allocation errors dynamically
            error_str = str(e).lower()
            if "not enough memory" in error_str or "out of memory" in error_str or isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                gc.collect()
                if attempt < max_retries - 1:
                    gr.Warning("OOM Error detected! Downsampling image by half and retrying...")
                    current_img = cv2.resize(current_img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
                    click_x = click_x // 2
                    click_y = click_y // 2
                else:
                    raise gr.Error("Out of Memory error persists even after downsampling. Please try a smaller image.")
            else:
                raise gr.Error(f"Unexpected Error: {str(e)}")
            
    return None, None, None
def zoom_on_click(evt: gr.SelectData):
    global current_fs_overlay, current_refined_overlay
    if current_fs_overlay is None or current_refined_overlay is None: 
        return None, None
    
    x, y = evt.index
    H, W, _ = current_fs_overlay.shape
    
    x1, x2 = max(0, x - 128), min(W, x + 128)
    y1, y2 = max(0, y - 128), min(H, y + 128)
    
    fs_crop = current_fs_overlay[y1:y2, x1:x2]
    ref_crop = current_refined_overlay[y1:y2, x1:x2]
    
    # UPSCALING for high DPI viewing in UI
    fs_crop_hr = cv2.resize(fs_crop, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    ref_crop_hr = cv2.resize(ref_crop, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    
    return fs_crop_hr, ref_crop_hr

# ==========================================
# 4. GRADIO UI LAYOUT
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🎯 FastSAM Interactive Boundary Refinement")
    gr.Markdown("**Step 1:** Upload an image and **click on the object** you want to segment.\n**Step 2:** Scroll down and **click anywhere on the left image** to zoom in.")
    
    # Top Row: Full Images
    with gr.Row():
        img_input = gr.Image(label="1. Original (Upload & Click to Segment)", interactive=True)
        fs_output = gr.Image(label="2. FastSAM Output", interactive=False)
        ref_output = gr.Image(label="3. Our Refined Output", interactive=False)
        
    gr.Markdown("---")
    
    # Bottom Row: Click target + Zoomed Patches
    with gr.Row(equal_height=True):
        # Column 1
        with gr.Column():
            gr.Markdown("### 🖱️ Click Here to Zoom")
            click_target = gr.Image(show_label=False, interactive=True) 
            
        # Column 2
        with gr.Column():
            gr.Markdown("### 🔍 Zoomed: FastSAM")
            zoom_fs = gr.Image(show_label=False, interactive=False, container=False)
            
        # Column 3
        with gr.Column():
            gr.Markdown("### 🔍 Zoomed: Refined")
            zoom_ref = gr.Image(show_label=False, interactive=False, container=False)

    # Event Wiring
    img_input.select(
        fn=segment_on_click, 
        inputs=img_input, 
        outputs=[fs_output, ref_output, click_target] 
    )
    
    click_target.select(
        fn=zoom_on_click, 
        inputs=None, 
        outputs=[zoom_fs, zoom_ref]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)