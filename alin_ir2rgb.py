import cv2
import numpy as np
import os
import re


def load_mask_gt(mask_path, frame_id):
    with open(mask_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if int(parts[0]) == frame_id:
                x, y, w, h = map(int, parts[2:6])
                return (x, y, w, h)
    return None

def load_det(det_path, frame_id):
    dets = []
    with open(det_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if int(parts[0]) == frame_id:
                x, y, w, h = map(float, parts[2:6])
                dets.append((x, y, w, h))
    return dets

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0
    return interArea / unionArea

def find_best_ir_box(mask_box, ir_dets):
    best_iou = 0
    best_box = None
    for det in ir_dets:
        score = iou(mask_box, det)
        if score > best_iou:
            best_iou = score
            best_box = det
    return best_box

def paste_patch(rgb_img, patch, bbox):
    x, y, w, h = bbox
    patch_resized = cv2.resize(patch, (w, h))
    rgb_img[y:y+h, x:x+w] = patch_resized
    return rgb_img

def extract_frame_id(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group()) if match else -1

def main():
    rgb_img_path = 'rgb_00004.jpg'  # 替换为你的输入
    ir_img_path = 'ir_000164.jpg'    # 替换为你的输入
    mask_gt_path = os.path.join('datasets', 'mask_0188gt.txt')
    ir_det_path = os.path.join('datasets', 'inf_det.txt')
    output_path = 'filled_000164.jpg'

    frame_id = extract_frame_id(rgb_img_path)

    rgb = cv2.imread(rgb_img_path)
    ir = cv2.imread(ir_img_path)

    mask_box = load_mask_gt(mask_gt_path, frame_id)
    ir_dets = load_det(ir_det_path, frame_id)

    if mask_box is None:
        print("No mask box found for this frame.")
        return

    best_ir_box = find_best_ir_box(mask_box, ir_dets)

    if best_ir_box is not None:
        dx, dy, dw, dh = map(int, best_ir_box)
        ir_patch = ir[dy:dy+dh, dx:dx+dw]
        rgb = paste_patch(rgb, ir_patch, mask_box)
        print("Used best matching IR det box for patch.")
    else:
        x, y, w, h = mask_box
        ir_patch = ir[y:y+h, x:x+w]  # fallback to same region
        rgb = paste_patch(rgb, ir_patch, mask_box)
        print("No matching IR box, used fallback region.")

    cv2.imwrite(output_path, rgb)
    print(f"Saved result to {output_path}")

if __name__ == '__main__':
    main()
