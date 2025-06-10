import os
import cv2
import numpy as np
import pandas as pd

# ====== 配置区域（可修改）======
rgb_gt_path = 'rgb_gt.txt'
ir_gt_path = 'ir_gt.txt'
mask_flag_path = 'mask_gt.txt'
H_path = 'H.npy'  # IR → RGB 的变换矩阵
rgb_img_dir = 'rgb_images'
output_dir = 'fused_output'

os.makedirs(output_dir, exist_ok=True)

# ====== 工具函数 ======
def load_mot_gt_by_frame(filepath):
    data = pd.read_csv(filepath, header=None)
    frame_dict = {}
    for _, row in data.iterrows():
        frame_id = int(row[0])
        tid = int(row[1])
        x, y, w, h = map(float, row[2:6])
        frame_dict.setdefault(frame_id, []).append((tid, x, y, w, h))
    return frame_dict

def load_masked_frames(filepath):
    masked_frames = set()
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if parts:
                masked_frames.add(int(parts[0]))
    return masked_frames

def draw_boxes(image, boxes, color, prefix='ID'):
    for tid, x, y, w, h in boxes:
        pt1 = (int(x), int(y))
        pt2 = (int(x + w), int(y + h))
        center = (int(x + w / 2), int(y + h))
        cv2.rectangle(image, pt1, pt2, color, 2)
        cv2.putText(image, f'{prefix}{tid}', (center[0]+5, center[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

def transform_ir_boxes(ir_boxes, H):
    transformed = []
    for tid, x, y, w, h in ir_boxes:
        pt = np.array([x + w / 2, y + h, 1.0]).reshape(3, 1)
        dst = H @ pt
        dst /= dst[2]
        cx, cy = dst[0][0], dst[1][0]
        transformed.append((tid, cx - w/2, cy - h, w, h))
    return transformed

# ====== 主流程 ======
def main():
    rgb_gt = load_mot_gt_by_frame(rgb_gt_path)
    ir_gt = load_mot_gt_by_frame(ir_gt_path)
    masked_frames = load_masked_frames(mask_flag_path)
    H = np.load(H_path)

    transformed_records = []

    for frame_id in sorted(masked_frames):
        rgb_path = os.path.join(rgb_img_dir, f'rgb_{frame_id:06d}.jpg')
        if not os.path.exists(rgb_path):
            print(f"[跳过] 未找到图像 {rgb_path}")
            continue

        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            continue

        rgb_boxes = rgb_gt.get(frame_id, [])
        ir_boxes = ir_gt.get(frame_id, [])

        rgb_img = draw_boxes(rgb_img, rgb_boxes, (0, 255, 0), prefix='RGB')

        if ir_boxes:
            transformed_ir_boxes = transform_ir_boxes(ir_boxes, H)
            rgb_img = draw_boxes(rgb_img, transformed_ir_boxes, (0, 0, 255), prefix='IR')

            for tid, x, y, w, h in transformed_ir_boxes:
                transformed_records.append((frame_id, tid, round(x, 2), round(y, 2), round(w, 2), round(h, 2)))

        cv2.imwrite(os.path.join(output_dir, f'fused_{frame_id:06d}.jpg'), rgb_img)

    # 保存转换后的IR框数据
    df = pd.DataFrame(transformed_records, columns=['frame', 'id', 'x', 'y', 'w', 'h'])
    df.to_csv(os.path.join(output_dir, 'ir_transformed_boxes.txt'), index=False, header=False)
    print(f"✅ 融合完成，图像与IR框保存至: {output_dir}")

if __name__ == '__main__':
    main()
