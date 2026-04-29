import os
import cv2
import numpy as np
import random
import torch
import gc
from ultralytics import FastSAM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return 0.0 if union == 0 else intersection / union

# --- PATHS & PARAMS ---
im_dir = r"D:\sems\AIP\Proj\dat\DIS-TR\im"
gt_dir = r"D:\sems\AIP\Proj\dat\DIS-TR\gt"
fs_out_dir = r"D:\sems\AIP\Proj\dat\DIS-TR\fastsam"
os.makedirs(fs_out_dir, exist_ok=True)

IOU_THRESHOLD = 0.9
AREA_TOLERANCE = 0.10
NUM_WORKERS = 10
MAX_IMAGE_DIM = 4500  # skip images larger than this in either height or width

class DIS5KPrefetchDataset(Dataset):
    def __init__(self, img_files):
        self.img_files = img_files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        base_name = img_name.rsplit('.', 1)[0]
        img_path = os.path.join(im_dir, img_name)
        gt_path = os.path.join(gt_dir, base_name + '.png')

        if not os.path.exists(gt_path):
            return None

        img = cv2.imread(img_path)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if img is None or gt_mask is None:
            return None

        # Skip very large images to reduce OOM risk
        if max(img.shape[0], img.shape[1]) > MAX_IMAGE_DIM:
            print(f"Skipped {base_name} due to large dimensions: {img.shape}")
            return None

        if len(gt_mask.shape) > 2:
            gt_mask = gt_mask[:, :, 0]

        gt_bin = (gt_mask > 127).astype(np.uint8)
        gt_area = gt_bin.sum()

        if gt_area == 0:
            return None

        y_indices, x_indices = np.where(gt_bin > 0)
        rand_idx = random.randint(0, len(y_indices) - 1)
        point = [int(x_indices[rand_idx]), int(y_indices[rand_idx])]

        return {
            'img_path': img_path,
            'img_shape': img.shape,
            'gt_bin': gt_bin,
            'gt_area': gt_area,
            'point': point,
            'base_name': base_name
        }

# Fix for Windows multiprocessing pickling error
def custom_collate_fn(batch):
    return batch

if __name__ == '__main__':
    print("Loading FastSAM model...")
    model = FastSAM("FastSAM-x.pt")

    img_files = [f for f in os.listdir(im_dir) if f.endswith('.jpg')]

    dataset = DIS5KPrefetchDataset(img_files)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn
    )

    saved_count = 0

    print(f"Processing {len(img_files)} images using {NUM_WORKERS} CPU cores...")
    for batch in tqdm(loader, desc="Generating Masks"):
        data = batch[0]
        if data is None:
            continue

        try:
            # FastSAM Prediction
            with torch.no_grad():
                results = model(
                    data['img_path'],
                    points=[data['point']],
                    labels=[1],
                    verbose=False
                )

            if results[0].masks is not None and len(results[0].masks.data) > 0:
                fs_mask_raw = results[0].masks.data.cpu().numpy()[0]
                fs_mask = cv2.resize(
                    fs_mask_raw,
                    (data['img_shape'][1], data['img_shape'][0]),
                    interpolation=cv2.INTER_NEAREST
                )
                fs_bin = (fs_mask > 0.5).astype(np.uint8)
                fs_area = fs_bin.sum()

                iou = compute_iou(fs_bin, data['gt_bin'])
                area_diff_ratio = abs(float(fs_area) - float(data['gt_area'])) / float(data['gt_area'])

                if iou >= IOU_THRESHOLD and area_diff_ratio <= AREA_TOLERANCE:
                    save_path = os.path.join(fs_out_dir, data['base_name'] + '.png')
                    cv2.imwrite(save_path, fs_bin * 255)
                    saved_count += 1

            del results

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\nSkipped {data['base_name']} due to OOM spike.")
            else:
                print(f"\nSkipped {data['base_name']} due to RuntimeError: {e}")
        except Exception as e:
            print(f"\nSkipped {data['base_name']} due to Error: {e}")

        # Force GC and clear cache
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nDone! Saved {saved_count} refined FastSAM masks.")