import torch
import torch.nn as nn

class IterativeBoundaryRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Image Features (32 channels)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 2. NEW: Mask Feature Extractor (16 channels)
        self.mask_stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 3. Looper: 32 (img) + 16 (mask) = 48 channels
        self.looper = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=3, padding=2, dilation=2), 
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, img, init_mask, iters=3):
        img_feats = self.stem(img)
        
        # FIX: Reduced from 6.0 to 2.0. 
        # Now the network can actually overpower FastSAM's initial guess!
        curr_logits = (init_mask - 0.5) * 2.0 

        for _ in range(iters):
            curr_mask = torch.sigmoid(curr_logits)
            
            # Extract features from the current mask state
            mask_feats = self.mask_stem(curr_mask)
            
            # Concatenate 32 img channels + 16 mask channels
            x = torch.cat([img_feats, mask_feats], dim=1)
            
            delta = self.looper(x)
            curr_logits = curr_logits + delta
            
        return torch.sigmoid(curr_logits)