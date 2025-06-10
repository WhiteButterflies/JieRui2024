# import cv2
# import numpy as np
# import os
# from collections import defaultdict
#
# # 用户配置路径
# rgb_gt_path = r'D:\JieRui2024\datasets\0061gt.txt'
# ir_gt_path = r'D:\JieRui2024\datasets\0061gt_mask.txt'
# save_path = '0061_ir_to_rgb_h.npy'
# frame_id = 1  # 默认使用第一帧配对
#
# # 解析 GT 文件
# def load_mot_gt(filepath, target_frame):
#     boxes = defaultdict(tuple)
#     with open(filepath, 'r') as f:
#         for line in f:
#             parts = line.strip().split(',')
#             try:
#                 frame = int(float(parts[0]))  # 支持 '1.0' 或 '1'
#                 if frame != target_frame:
#                     continue
#                 tid = int(float(parts[1]))
#                 x, y, w, h = map(float, parts[2:6])
#                 cx = x + w / 2
#                 cy = y + h / 2
#                 boxes[tid] = (cx, cy)
#             except ValueError as e:
#                 print(f"跳过无法解析的行: {line.strip()} 错误: {e}")
#     return boxes
#
# # 匹配 RGB/IR 中相同 ID 的目标
# def match_targets(rgb_dict, ir_dict):
#     matched_rgb = []
#     matched_ir = []
#     for tid in rgb_dict:
#         if tid in ir_dict:
#             matched_rgb.append(rgb_dict[tid])
#             matched_ir.append(ir_dict[tid])
#     return matched_rgb, matched_ir
#
# # 主流程
# def main():
#     rgb_centers = load_mot_gt(rgb_gt_path, frame_id)
#     ir_centers = load_mot_gt(ir_gt_path, frame_id)
#
#     print(f"RGB IDs: {list(rgb_centers.keys())}")
#     print(f" IR IDs: {list(ir_centers.keys())}")
#
#     rgb_pts, ir_pts = match_targets(rgb_centers, ir_centers)
#
#     if len(rgb_pts) < 3:
#         print("匹配点数不足，无法估算单应性矩阵（需要至少3对）。")
#         return
#
#     rgb_pts_np = np.array(rgb_pts, dtype=np.float32)
#     ir_pts_np = np.array(ir_pts, dtype=np.float32)
#
#     H, mask = cv2.findHomography(ir_pts_np, rgb_pts_np, method=cv2.RANSAC)
#     if H is not None:
#         np.save(save_path, H)
#         print(f"成功保存单应性矩阵到：{save_path}")
#         print(f"共匹配 ID 数：{len(rgb_pts)}")
#     else:
#         print("计算单应性矩阵失败。")
#
# if __name__ == '__main__':
#     main()
import cv2
import numpy as np
import os

# ==== 可调参数 ====
NUM_FRAMES = 5
AREA_THRESH = 4000
MAX_MATCHED_POINTS = 10
FRAME_START = 1
VISUALIZE = True  # 是否保存每帧可视化图像
SHOW_IMAGE = False  # 是否用cv2.imshow显示每帧（调试时可以用）

# ==== 文件路径 ====
rgb_gt_path = r'D:\JieRui2024\datasets\rgb_0275_gt.txt'
ir_gt_path = r'D:\JieRui2024\datasets\inf_0275_gt.txt'
mask_flag_path = r'D:\JieRui2024\datasets\mask_0275gt.txt'
save_path = '0275_ir_to_rgb_h.npy'

# 图像路径模板
rgb_img_template = r'D:\5-16data\jierui24_final_RGB\train\0275\image\{:06d}.jpg'
ir_img_template = r'D:\5-16data\jierui24_final_RGB\train\0275\image\{:06d}.jpg'
def load_mask_flags(mask_path):
    masked_frames = set()
    with open(mask_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if int(parts[0]) == 1:
                frame_id = int(parts[1])
                masked_frames.add(frame_id)
    return masked_frames


def load_gt_dict_per_frame(filepath):
    gt_per_frame = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            try:
                frame = int(float(parts[0]))
                tid = int(float(parts[1]))
                x, y, w, h = map(float, parts[2:6])
                if frame not in gt_per_frame:
                    gt_per_frame[frame] = {}
                gt_per_frame[frame][tid] = (x, y, w, h)
            except:
                continue
    return gt_per_frame


def draw_boxes_and_matches(img, all_boxes, matched_pts_dict, frame_id):
    for tid, (x, y, w, h) in all_boxes.items():
        color = (0, 255, 0)
        if tid in matched_pts_dict:
            color = (0, 0, 255)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        if tid in matched_pts_dict:
            cx, cy = matched_pts_dict[tid]
            cv2.circle(img, (int(cx), int(cy)), 5, color, -1)
            cv2.putText(img, f'{tid}', (int(cx + 5), int(cy - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(img, f'Frame {frame_id}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img


def main():
    rgb_gt = load_gt_dict_per_frame(rgb_gt_path)
    ir_gt = load_gt_dict_per_frame(ir_gt_path)
    masked_frames = load_mask_flags(mask_flag_path)

    matched_rgb = []
    matched_ir = []
    frames_checked = 0
    current_frame = FRAME_START

    while frames_checked < NUM_FRAMES:
        if current_frame in masked_frames:
            current_frame += 1
            continue

        rgb_objs = rgb_gt.get(current_frame, {})
        ir_objs = ir_gt.get(current_frame, {})

        frame_matched_rgb = {}
        frame_matched_ir = {}

        for tid, (x, y, w, h) in rgb_objs.items():
            area = w * h
            if tid in ir_objs and area < AREA_THRESH:
                # 计算底部中心点
                bx_rgb = x + w / 2
                by_rgb = y + h
                x2, y2, w2, h2 = ir_objs[tid]
                bx_ir = x2 + w2 / 2
                by_ir = y2 + h2

                matched_rgb.append((bx_rgb, by_rgb))
                matched_ir.append((bx_ir, by_ir))
                frame_matched_rgb[tid] = (bx_rgb, by_rgb)
                frame_matched_ir[tid] = (bx_ir, by_ir)

                if len(matched_rgb) >= MAX_MATCHED_POINTS:
                    break

        # 可视化
        if VISUALIZE:
            rgb_img_path = rgb_img_template.format(current_frame)
            rgb_img = cv2.imread(rgb_img_path)
            if rgb_img is not None:
                vis = draw_boxes_and_matches(rgb_img.copy(), rgb_objs, frame_matched_rgb, current_frame)
                vis_save_path = f'vis_frame_{current_frame:04d}.jpg'
                cv2.imwrite(vis_save_path, vis)
                if SHOW_IMAGE:
                    cv2.imshow('vis', vis)
                    cv2.waitKey(500)

        if len(matched_rgb) >= MAX_MATCHED_POINTS:
            break

        frames_checked += 1
        current_frame += 1

    if SHOW_IMAGE:
        cv2.destroyAllWindows()

    if len(matched_rgb) < 3:
        print(f"有效匹配点不足（仅找到 {len(matched_rgb)} 对），无法生成单应性矩阵。")
        return

    rgb_pts = np.array(matched_rgb, dtype=np.float32).reshape(-1, 2)
    ir_pts = np.array(matched_ir, dtype=np.float32).reshape(-1, 2)

    H, mask = cv2.findHomography(ir_pts, rgb_pts, method=cv2.RANSAC)
    if H is not None:
        np.save(save_path, H)
        print(f"单应性矩阵保存成功：{save_path}")
        print(f"共使用 {len(rgb_pts)} 对匹配点")
    else:
        print("单应性矩阵估计失败。")


if __name__ == '__main__':
    main()