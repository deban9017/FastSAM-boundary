import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random

from model import IterativeBoundaryRefiner
from loss import SoftBoundaryLoss 
from dataset import PreCroppedPatchDataset

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    model = IterativeBoundaryRefiner().to(device)
    criterion = SoftBoundaryLoss(blur_kernel=11)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Reading directly from dat_patch
    train_dataset = PreCroppedPatchDataset('dat_patch/train', 'dat_patch/train_masks')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
    
    epochs = 10
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (imgs, fs_masks, gt_masks) in enumerate(train_loader):
            imgs = imgs.to(device)
            fs_masks = fs_masks.to(device)
            gt_masks = gt_masks.to(device)
            
            optimizer.zero_grad()
            iters = random.randint(2, 5)
            refined_masks = model(imgs, fs_masks, iters)
            
            loss = criterion(refined_masks, gt_masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(train_loader)
        print(f"--- Epoch {epoch} Avg Loss: {avg_loss:.4f} ---")
        scheduler.step(avg_loss)
        
        torch.save(model.state_dict(), f'code_nn/RCNN/refiner_epoch_{epoch}.pth')