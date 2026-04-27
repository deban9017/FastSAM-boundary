import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftBoundaryLoss(nn.Module):
    def __init__(self, blur_kernel=31):
        super().__init__()
        self.blur_kernel = blur_kernel

    def extract_boundary(self, mask):
        dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        return dilated - eroded

    def forward(self, pred, target):
        gt_boundary = self.extract_boundary(target)
        
        soft_glow = F.avg_pool2d(gt_boundary, kernel_size=self.blur_kernel, stride=1, padding=self.blur_kernel//2)
        soft_glow = soft_glow / (soft_glow.max() + 1e-6)
        weight_map = 0.2 + (4.8 * soft_glow)
        
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        weighted_loss = bce_loss * weight_map
        
        return weighted_loss.mean()