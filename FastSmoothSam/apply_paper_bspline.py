import os
import glob
import cv2
import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev
from skimage.feature import canny
from skimage.color import rgb2gray

# ---------------------------------------------------------
# Official FastSmoothSAM Helper Functions
# ---------------------------------------------------------

def skimage_canny(image, sigma):
    img_gray = rgb2gray(image)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    edges = canny(img_gray, sigma)
    edge_points = np.column_stack(np.where(edges > 0))  
    return np.array(edge_points)

def linear_interpolation(point1, point2, n):
    x1, y1 = point1
    x2, y2 = point2
    step_size = 1 / n
    return [(x1 + i * step_size * (x2 - x1), y1 + i * step_size * (y2 - y1)) for i in range(n + 1)]

def interpolate_mask_edge_points(mask_edge_points):
    inter_mask_edge_points = []
    total_num_points = len(mask_edge_points)
    for i in range(total_num_points):
        current_point = np.array(mask_edge_points[i])
        next_point = np.array(mask_edge_points[(i + 1) % total_num_points])
        distance = np.linalg.norm(current_point - next_point)
        inter_points_num = max(int(distance / 4), 1)
        interpolated_points = linear_interpolation(current_point, next_point, inter_points_num)
        inter_mask_edge_points.extend(interpolated_points[:-1])
    return np.array(inter_mask_edge_points)

def unique_datas(points_data):
    unique_data, unique_indices = np.unique(points_data, axis=0, return_index=True)
    return unique_data[np.argsort(unique_indices)]

def AdS(points, curvatures, threshold):
    points = points.T
    filtered_indices = curvatures > threshold
    filtered_points = points[:, filtered_indices]
    return filtered_points.T

def b_spline(points, k, node, num_fit):
    points = unique_datas(points)
    if len(points) <= k:
        return None, None, None, None
    try:
        x_values = points[:, 0]
        y_values = points[:, 1]
        
        if num_fit == 1:
            tck, u = splprep([x_values, y_values], k=k, per=True)
            u_new = np.linspace(u.min(), u.max(), node)
            out = splev(u_new, tck)
            dx, dy = splev(u_new, tck, der=1)
            d2x, d2y = splev(u_new, tck, der=2)
            curvatures = np.abs(dx * d2y - dy * d2x) / (dx ** 2 + dy ** 2) ** 1.5
        elif num_fit == 2:
            tck, u = splprep([x_values, y_values], k=k, s=50, per=True)
            u_new = np.linspace(u.min(), u.max(), node)
            out = splev(u_new, tck)
            curvatures = None
        else:
            return None, None, None, None
            
        fit_points = np.column_stack(out)
        return fit_points, out[0], out[1], curvatures
    except Exception as e:
        return None, None, None, None

def match_dilated_boundary(boundary, ori_edge_points, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_boundary = cv2.dilate(boundary, kernel, iterations=1)
    height, width = dilated_boundary.shape
    edge_points = np.array(ori_edge_points)
    
    valid_mask = (edge_points[:, 0] >= 0) & (edge_points[:, 0] < height) & \
                 (edge_points[:, 1] >= 0) & (edge_points[:, 1] < width)
    valid_edge_points = edge_points[valid_mask]
    
    final_valid_mask = dilated_boundary[valid_edge_points[:, 0], valid_edge_points[:, 1]] > 0
    obj_edge_points = valid_edge_points[final_valid_mask]
    
    obj_edge_points = [(y, x) for x, y in obj_edge_points] 
    if len(obj_edge_points) == 0:
        return None
    return np.array(obj_edge_points)

def get_thin_points(point_set, fit_points, sample_points, radius):
    points_num = len(fit_points)
    point_set = np.array(point_set)
    sample_points = set(map(tuple, sample_points))
    kdtree_all = KDTree(point_set)
    kdtree_fit = KDTree(fit_points)
    thin_points = []
    processed_points = set()
    query_cache = {}

    for i, point in enumerate(point_set):
        if i > points_num: break
        point_tuple = tuple(point)
        if point_tuple in processed_points: continue
        
        if point_tuple not in query_cache:
            query_cache[point_tuple] = kdtree_all.query_ball_point(point, radius)
        points_indices_all = query_cache[point_tuple]
        points_in_radius = [point_set[idx] for idx in points_indices_all]

        if point_tuple in sample_points:
            if point_tuple not in query_cache:
                query_cache[point_tuple] = kdtree_fit.query_ball_point(point, radius)
            points_indices_fit = query_cache[point_tuple]
            points_in_radius_fit = [point_set[idx] for idx in points_indices_fit]
            
            set1 = set(map(tuple, points_in_radius))
            set2 = set(map(tuple, points_in_radius_fit))
            difference_set = set1 - set2
            if len(difference_set) > 0:
                result_points = np.array(list(difference_set))
                points_in_radius = [np.array(row, dtype='int64') for row in result_points]
        else:
            for idx in points_indices_all:
                processed_points.add(tuple(point_set[idx]))
                
        if points_in_radius:
            center = tuple(map(lambda x: sum(x) / len(x), zip(*points_in_radius)))
            thin_points.append(center)
            
    return thin_points

# ---------------------------------------------------------
# Pipeline Orchestration
# ---------------------------------------------------------

def sequence_points_knn(points):
    """
    Forces top-to-bottom unordered Canny points back into a continuous 
    perimeter sequence using a nearest-neighbor walk.
    """
    if len(points) <= 1:
        return points
        
    points_list = points.tolist()
    ordered = [points_list.pop(0)] # Start at an arbitrary point
    
    while points_list:
        last_pt = np.array(ordered[-1])
        rem_pts = np.array(points_list)
        
        # Find the index of the closest remaining point
        dists = np.linalg.norm(rem_pts - last_pt, axis=1)
        closest_idx = np.argmin(dists)
        
        # Pop it from remaining and add to ordered
        ordered.append(points_list.pop(closest_idx))
        
    return np.array(ordered)

def apply_fastsmoothsam(image, mask, radius=5, curve_thresh=0.03):
    # Step 0: Extract base contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return mask
    contour = max(contours, key=cv2.contourArea).squeeze()
    
    if len(contour.shape) < 2 or len(contour) < 10: 
        return mask
        
    # Interpolate for baseline curve definition
    inter_points = interpolate_mask_edge_points(contour)
    if len(inter_points) < 4: return mask
    
    # STAGE 1: Coarse Fitting
    fit_points, _, _, curvatures = b_spline(inter_points, k=3, node=len(inter_points), num_fit=1)
    if fit_points is None or curvatures is None: return mask
    
    # STAGE 2: Canny Edge Matching
    canny_edge_points = skimage_canny(image, sigma=1.0)
    boundary_img = np.zeros(mask.shape, dtype=np.uint8)
    cv2.polylines(boundary_img, [fit_points.astype(np.int32)], True, 255, 1)
    obj_edge_points = match_dilated_boundary(boundary_img, canny_edge_points, kernel_size=radius*2)
    
    final_contour = fit_points.astype(np.int32)
    
# STAGE 3 & 4: Adaptive Sampling, Thinning, and Fine Fitting
    if obj_edge_points is not None and len(obj_edge_points) > 0:
        sample_points = AdS(fit_points, curvatures, threshold=curve_thresh)
        thin_points = get_thin_points(obj_edge_points, fit_points, sample_points, radius=radius)
        
        if thin_points and len(thin_points) > 3:
            thin_points_arr = np.array(thin_points)
            
            # --- THE FIX ---
            # Restitch the broken top-down points into a clean perimeter loop
            thin_points_arr = sequence_points_knn(thin_points_arr)
            # ---------------

            fine_fit_points, _, _, _ = b_spline(thin_points_arr, k=2, node=len(contour)*2, num_fit=2)
            
            if fine_fit_points is not None:
                final_contour = fine_fit_points.astype(np.int32)
            else:
                final_contour = thin_points_arr.astype(np.int32)

    # Draw final smoothed mask
    smoothed_mask = np.zeros_like(mask)
    cv2.fillPoly(smoothed_mask, [final_contour], 255)
    return smoothed_mask




# ---------------------------------------------------------
# Dataset Processing Loop
# ---------------------------------------------------------

def process_dataset(base_path="dat_patch"):
    splits = ['test']
    
    for split in splits:
        print(f"Processing {split} set...")
        img_dir = os.path.join(base_path, split)
        mask_dir = os.path.join(base_path, f"{split}_masks", "fastsam")
        out_dir = os.path.join(base_path, f"{split}_masks", "fastsmoothsam_paper")
        os.makedirs(out_dir, exist_ok=True)
        
        mask_paths = glob.glob(os.path.join(mask_dir, "*.png"))
        
        for mask_path in mask_paths:
            filename = os.path.basename(mask_path)
            base_name = os.path.splitext(filename)[0]
            img_path = os.path.join(img_dir, base_name + ".jpg")
            
            if not os.path.exists(img_path):
                print(f"  [Warning] Original image not found: {img_path}")
                continue
                
            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None or image is None:
                continue
                
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # Padding to prevent edge-of-frame interpolation errors
            mask_padded = cv2.copyMakeBorder(mask, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
            image_padded = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
            
            smoothed_mask_padded = apply_fastsmoothsam(image_padded, mask_padded)
            smoothed_mask = smoothed_mask_padded[10:-10, 10:-10]
            
            out_path = os.path.join(out_dir, filename)
            cv2.imwrite(out_path, smoothed_mask)
            
        print(f"Finished {split} set. Saved to {out_dir}\n")

if __name__ == "__main__":
    process_dataset("dat_patch")