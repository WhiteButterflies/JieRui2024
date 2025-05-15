import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def detect_black_boxes(img, area_thresh=500, black_thresh=30):
    # 读取图像（确保3通道）
    if img is None:
        raise ValueError("Image not found or invalid path")

    # 方法1：直接在BGR空间检测纯黑（更严格）
    # black_mask = np.all(img < [black_thresh] * 3, axis=2).astype(np.uint8) * 255

    # 方法2（备选）：灰度图+阈值（适用于灰度场景）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, black_mask = cv2.threshold(gray, black_thresh, 255, cv2.THRESH_BINARY_INV)

    # 去噪（开运算）
    # kernel = np.ones((3, 3), np.uint8)
    # black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    vis_img = img.copy()
    for cnt in contours:
        # 使用最小外接矩形（更精确）
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(int)
        area = cv2.contourArea(cnt)

        if area > area_thresh:
            boxes.append(box)
            cv2.drawContours(vis_img, [box], 0, (0, 0, 255), 2)  # 绘制红色旋转矩形

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


