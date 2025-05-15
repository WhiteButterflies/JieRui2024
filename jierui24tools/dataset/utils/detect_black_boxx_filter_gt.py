import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


# def detect_black_boxes(img, area_thresh=500, black_thresh=30):
#     """
#     Detect black boxes in an image and return them as (x, y, width, height) rectangles.
#
#     Args:
#         img: Input BGR image
#         area_thresh: Minimum area threshold for boxes (default: 500)
#         black_thresh: Maximum intensity value to consider as black (0-255, default: 30)
#
#     Returns:
#         List of bounding boxes in (x, y, w, h) format
#     """
#     if img is None:
#         raise ValueError("Image not found or invalid path")
#
#     # Convert to grayscale and threshold
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, black_mask = cv2.threshold(gray, black_thresh, 255, cv2.THRESH_BINARY_INV)
#
#     # Find contours
#     contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     boxes = []
#     for cnt in contours:
#         # Get axis-aligned bounding rectangle
#         x, y, w, h = cv2.boundingRect(cnt)
#         area = w * h
#
#         if area > area_thresh:
#             boxes.append((x, y, w, h))
#
#     return boxes

def detect_black_boxes(img, area_thresh=500):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > area_thresh:
            boxes.append((x, y, w, h))
    return boxes

def is_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    inter_w = max(0, min(x1_max, x2_max) - max(x1, x2))
    inter_h = max(0, min(y1_max, y2_max) - max(y1, y2))
    return inter_w > 0 and inter_h > 0

def load_gt(gt_path):
    cols = ["frame", "id", "x", "y", "w", "h", "conf", "class", "vis"]
    return pd.read_csv(gt_path, header=None, names=cols)

def process_sequence(image_dir, gt_path, output_gt_path):
    gt_df = load_gt(gt_path)

    # 预处理：按帧编号组织GT
    frame_dict = {fid: df for fid, df in gt_df.groupby("frame")}
    filtered_rows = []

    total_frames = len([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
    for frame_id in tqdm(range(1, total_frames + 1)):
        filename = os.path.join(image_dir, f"{frame_id:06d}.jpg")
        if not os.path.exists(filename):
            continue

        img = cv2.imread(filename)
        black_boxes = detect_black_boxes(img)

        frame_gt = frame_dict.get(frame_id, pd.DataFrame())
        for _, row in frame_gt.iterrows():
            gt_box = (row["x"], row["y"], row["w"], row["h"])
            if not any(is_overlap(gt_box, b) for b in black_boxes):
                filtered_rows.append(row)

    # 保存结果
    filtered_df = pd.DataFrame(filtered_rows)
    filtered_df.to_csv(output_gt_path, header=False, index=False)


