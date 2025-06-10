import cv2
import numpy as np
import os
import re

H_PATH = '0061_ir_to_rgb_h_maxmatch.npy'  # 保存的单应性矩阵路径

def extract_frame_id(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group()) if match else -1

def load_mask_gt(mask_path, frame_id):
    with open(mask_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if int(parts[0]) == frame_id:
                x, y, w, h = map(int, parts[2:6])
                return (x, y, w, h)
    return None

def paste_patch(rgb_img, patch, bbox):
    x, y, w, h = bbox
    patch_resized = cv2.resize(patch, (w, h))
    rgb_img[y:y+h, x:x+w] = patch_resized
    return rgb_img

def apply_homography_to_ir(ir_img, h_matrix, target_shape):
    return cv2.warpPerspective(ir_img, h_matrix, (target_shape[1], target_shape[0]))

def main():
    # 修改此处以适配你要处理的图像
    rgb_img_path = 'rgb_000164.jpg'
    ir_img_path = 'ir_000164.jpg'
    mask_gt_path = os.path.join('D:\JieRui2024\datasets', 'mask_0061gt.txt')
    output_path = '5-29filled_0061_000164.jpg'

    frame_id = extract_frame_id(rgb_img_path)
    rgb = cv2.imread(rgb_img_path)
    ir = cv2.imread(ir_img_path)

    if rgb is None or ir is None:
        print("图像读取失败，请检查路径。")
        return

    # 加载遮挡框
    mask_box = load_mask_gt(mask_gt_path, frame_id)
    if mask_box is None:
        print("未找到当前帧的遮挡框。")
        return

    # 加载单应性矩阵
    if not os.path.exists(H_PATH):
        print(f"请先运行标定工具，生成单应性矩阵文件：{H_PATH}")
        return

    H = np.load(H_PATH)
    ir_aligned = apply_homography_to_ir(ir, H, rgb.shape)

    # 裁剪对齐后的 IR 图并补全到 RGB 遮挡区域
    x, y, w, h = mask_box
    ir_patch = ir_aligned[y:y+h, x:x+w]
    rgb_filled = paste_patch(rgb, ir_patch, mask_box)

    cv2.imwrite(output_path, rgb_filled)
    print(f"补全结果已保存至：{output_path}")

if __name__ == '__main__':
    main()
