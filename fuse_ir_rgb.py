#!/usr/bin/env python3
"""
fuse_ir_rgb.py

Usage:
    python fuse_ir_rgb.py --rgb_dir /path/to/rgb_images \
                         --ir_dir  /path/to/ir_images \
                         --output_dir /path/to/output \
                         [--threshold 10]

For each image in --rgb_dir matching *.jpg or *.png, 
look for a same-named file in --ir_dir, compute homography, 
warp IR to RGB frame, create mask of dark occlusion in RGB,
fuse the warped IR into occluded regions, and save side-by-side comparison.
"""
import os
import cv2
import numpy as np
import argparse

def compute_homography(rgb_img, ir_img):
    gray_rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    gray_ir = ir_img.copy() if len(ir_img.shape)==2 else cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
    # Feature detection
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(gray_rgb, None)
    kp2, des2 = orb.detectAndCompute(gray_ir, None)
    # Match with FLANN + Lowe's ratio test
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) < 4:
        raise RuntimeError("Not enough good matches found: %d" % len(good))
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    if H is None:
        raise RuntimeError("Homography estimation failed")
    return H

def fuse_and_save(rgb_path, ir_path, out_path, threshold):
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        print(f"[WARN] Cannot read RGB image: {rgb_path}")
        return
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        print(f"[WARN] Cannot read IR image: {ir_path}")
        return

    h, w = rgb.shape[:2]
    H = compute_homography(rgb, ir)
    warped_ir = cv2.warpPerspective(ir, H, (w, h))

    # Create occlusion mask from RGB dark regions
    gray_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    mask = (gray_rgb < threshold).astype(np.uint8) * 255
    # Morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_bool = mask.astype(bool)

    # Fuse IR into RGB
    fused = rgb.copy()
    for c in range(3):
        fused[:,:,c][mask_bool] = warped_ir[mask_bool]

    # Create side-by-side comparison
    vis_ir = cv2.cvtColor(warped_ir, cv2.COLOR_GRAY2BGR)
    comp = np.hstack([rgb, vis_ir, fused])

    # Save output
    cv2.imwrite(out_path, comp)
    print(f"[OK] Saved comparison: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Fuse IR into occluded RGB via homography")
    parser.add_argument("--rgb_dir", required=True, help="Directory of RGB images")
    parser.add_argument("--ir_dir",  required=True, help="Directory of IR images (matching filenames)")
    parser.add_argument("--output_dir", required=True, help="Directory to save fused comparisons")
    parser.add_argument("--threshold", type=int, default=10, help="Gray threshold for occlusion mask")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    exts = (".jpg", ".png", ".jpeg", ".bmp")
    for fn in sorted(os.listdir(args.rgb_dir)):
        if fn.lower().endswith(exts):
            rgb_path = os.path.join(args.rgb_dir, fn)
            ir_path  = os.path.join(args.ir_dir, fn)
            if not os.path.exists(ir_path):
                print(f"[WARN] Missing IR for {fn}, skipping")
                continue
            out_path = os.path.join(args.output_dir, fn)
            try:
                fuse_and_save(rgb_path, ir_path, out_path, args.threshold)
            except Exception as e:
                print(f"[ERROR] {fn}: {e}")

if __name__ == "__main__":
    main()
