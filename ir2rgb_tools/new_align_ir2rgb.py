# import cv2
# import numpy as np
# import os
#
# # ==== 可调参数 ====
# TARGET_MATCHED_POINTS = 120
# AREA_THRESH = 400000
# FRAME_START = 1
# FRAME_STEP = 1
# NUM_TRIES = 300
# SEARCH_DIRECTION = 'right'  # 'left' or 'right'
#
# # ==== 文件路径 ====
# rgb_gt_path = r'D:\JieRui2024\datasets\rgb_0061_gt.txt'
# ir_gt_path = r'D:\JieRui2024\datasets\inf_0061_gt.txt'
# mask_flag_path = r'D:\JieRui2024\datasets\mask_0061gt.txt'
# save_path = r'D:\JieRui2024\ir2rgb_tools\0061_affine_matrix.npy'
#
#
# log_path = '0275_matched_points_log.txt'
#
# rgb_img_template = r'D:\5-16data\jierui24_final_RGB\train\0275\image\{:06d}.jpg'
# ir_img_template = r'D:\5-16data\jierui24_final_INF\train\0275\image\{:06d}.jpg'
#
#
# def load_mask_flags(mask_path):
#     """
#     所有出现在 mask.txt 文件中的帧都被认为是被遮挡帧
#     """
#     masked = set()
#     with open(mask_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split(',')
#             if len(parts) >= 1:
#                 frame_id = int(parts[0])
#                 masked.add(frame_id)
#     return masked
#
#
# def load_gt_dict(filepath):
#     """
#     加载 GT 文件并按帧分类，筛除大目标
#     """
#     gt = {}
#     with open(filepath, 'r') as f:
#         for line in f:
#             parts = line.strip().split(',')
#             try:
#                 frame = int(float(parts[0]))
#                 tid = int(float(parts[1]))
#                 x, y, w, h = map(float, parts[2:6])
#                 if w * h < AREA_THRESH:
#                     if frame not in gt:
#                         gt[frame] = []
#                     gt[frame].append((tid, x, y, w, h))
#             except:
#                 continue
#     return gt
#
#
# def get_closest_target(boxes, center_x, direction):
#     """
#     在指定方向中找出离中心最近的目标，返回 (tid, bx, by)
#     """
#     best_box = None
#     min_dist = float('inf')
#     for tid, x, y, w, h in boxes:
#         cx = x + w / 2
#         by = y + h
#         if (direction == 'left' and cx < center_x) or (direction == 'right' and cx > center_x):
#             dist = abs(cx - center_x)
#             if dist < min_dist:
#                 min_dist = dist
#                 best_box = (tid, cx, by)
#     return best_box
#
#
# def draw_matched_point(img, cx, cy, tid, pid, color=(0, 0, 255)):
#     """
#     在图像上标注匹配点和目标 ID
#     """
#     cv2.circle(img, (int(cx), int(cy)), 6, color, -1)
#     cv2.putText(img, f'P{pid}-ID{tid}', (int(cx) + 5, int(cy) - 5),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#
#
# def main():
#     rgb_gt = load_gt_dict(rgb_gt_path)
#     ir_gt = load_gt_dict(ir_gt_path)
#     mask_flags = load_mask_flags(mask_flag_path)
#
#     matched_rgb = []
#     matched_ir = []
#
#     frame_id = FRAME_START
#     tries = 0
#     point_idx = 1
#
#     with open(log_path, 'w') as log_f:
#         log_f.write("Matched Points Log\n====================\n")
#
#         while len(matched_rgb) < TARGET_MATCHED_POINTS and tries < NUM_TRIES:
#             if frame_id in mask_flags:
#                 print(f"跳过被遮挡帧: {frame_id}")
#                 frame_id += FRAME_STEP
#                 tries += 1
#                 continue
#
#             rgb_objs = rgb_gt.get(frame_id, [])
#             ir_objs = ir_gt.get(frame_id, [])
#             if not rgb_objs or not ir_objs:
#                 frame_id += FRAME_STEP
#                 tries += 1
#                 continue
#
#             rgb_img_path = rgb_img_template.format(frame_id)
#             ir_img_path = ir_img_template.format(frame_id)
#             rgb_img = cv2.imread(rgb_img_path)
#             ir_img = cv2.imread(ir_img_path)
#             if rgb_img is None or ir_img is None:
#                 frame_id += FRAME_STEP
#                 tries += 1
#                 continue
#
#             rgb_center_x = rgb_img.shape[1] // 2
#             ir_center_x = ir_img.shape[1] // 2
#
#             rgb_target = get_closest_target(rgb_objs, rgb_center_x, SEARCH_DIRECTION)
#             ir_target = get_closest_target(ir_objs, ir_center_x, SEARCH_DIRECTION)
#
#             if rgb_target and ir_target:
#                 tid_rgb, bx_rgb, by_rgb = rgb_target
#                 tid_ir, bx_ir, by_ir = ir_target
#                 matched_rgb.append((bx_rgb, by_rgb))
#                 matched_ir.append((bx_ir, by_ir))
#
#                 log_f.write(f"Matched Point P{point_idx}:\n")
#                 log_f.write(f"  Frame: {frame_id}\n")
#                 log_f.write(f"  RGB: id={tid_rgb}, (x={bx_rgb:.2f}, y={by_rgb:.2f})\n")
#                 log_f.write(f"  IR : id={tid_ir}, (x={bx_ir:.2f}, y={by_ir:.2f})\n\n")
#
#                 draw_matched_point(rgb_img, bx_rgb, by_rgb, tid_rgb, point_idx, color=(0, 0, 255))
#                 draw_matched_point(ir_img, bx_ir, by_ir, tid_ir, point_idx, color=(255, 0, 0))
#                 #
#                 # cv2.imwrite(f"rgb_frame_{frame_id:06d}_matched.jpg", rgb_img)
#                 # cv2.imwrite(f"ir_frame_{frame_id:06d}_matched.jpg", ir_img)
#
#                 point_idx += 1
#
#             frame_id += FRAME_STEP
#             tries += 1
#
#     if len(matched_rgb) < 3:
#         print(f"❌ 匹配点不足（{len(matched_rgb)}），无法生成单应性矩阵。")
#         return
#
#     rgb_pts = np.array(matched_rgb, dtype=np.float32).reshape(-1, 2)
#     ir_pts = np.array(matched_ir, dtype=np.float32).reshape(-1, 2)
#
#     H, mask = cv2.findHomography(ir_pts, rgb_pts, method=cv2.RANSAC)
#     if H is not None:
#         np.save(save_path, H)
#         print(f"✅ 单应性矩阵保存成功: {save_path}")
#         print(f"共使用匹配点: {len(rgb_pts)}")
#     else:
#         print("❌ 单应性矩阵计算失败。")
#
#
# if __name__ == '__main__':
#     main()


import os
import cv2
import numpy as np

# ==== 可调参数 ====
TARGET_MATCHED_POINTS = 200
AREA_THRESH = 400000
SEARCH_DIRECTION = 'left'  # 'left' or 'right'

# ==== 文件路径 ====
rgb_gt_path = r'D:\JieRui2024\datasets\rgb_0275_gt.txt'
ir_gt_path = r'D:\JieRui2024\datasets\inf_0275_gt.txt'
mask_flag_path = r'D:\JieRui2024\datasets\mask_0275gt.txt'
save_path = r'D:\JieRui2024\jierui24tools\merge_INF2RGB_GT\best_affine\0275_affine_matrix.npy'
log_path = '0275_matched_points_log.txt'


def load_mask_flags(mask_path):
    masked = set()
    with open(mask_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 1:
                frame_id = int(parts[0])
                masked.add(frame_id)
    return masked


def load_gt_dict(filepath):
    gt = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            try:
                frame = int(float(parts[0]))
                tid = int(float(parts[1]))
                x, y, w, h = map(float, parts[2:6])
                if w * h < AREA_THRESH:
                    if frame not in gt:
                        gt[frame] = []
                    gt[frame].append((tid, x, y, w, h))
            except:
                continue
    return gt


def get_closest_target(boxes, center_x, direction):
    best_box = None
    min_dist = float('inf')
    for tid, x, y, w, h in boxes:
        cx = x + w / 2
        by = y + h
        if (direction == 'left' and cx < center_x) or (direction == 'right' and cx > center_x):
            dist = abs(cx - center_x)
            if dist < min_dist:
                min_dist = dist
                best_box = (tid, cx, by)
    return best_box


def main():
    rgb_gt = load_gt_dict(rgb_gt_path)
    ir_gt = load_gt_dict(ir_gt_path)
    mask_flags = load_mask_flags(mask_flag_path)

    matched_rgb = []
    matched_ir = []
    point_idx = 1

    # 获取最大帧号
    all_frame_ids = set()
    with open(rgb_gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 1:
                try:
                    frame = int(float(parts[0]))
                    all_frame_ids.add(frame)
                except:
                    continue
    max_frame_id = max(all_frame_ids)
    print(f"📌 从 RGB GT 中检测到最大帧号: {max_frame_id}")

    with open(log_path, 'w') as log_f:
        log_f.write("Matched Points Log\n====================\n")

        for frame_id in range(1, max_frame_id + 1):
            if len(matched_rgb) >= TARGET_MATCHED_POINTS:
                break

            if frame_id in mask_flags:
                continue

            rgb_objs = rgb_gt.get(frame_id, [])
            ir_objs = ir_gt.get(frame_id, [])
            if not rgb_objs or not ir_objs:
                continue

            # 直接使用图像中心点（假定图像宽度为1920）
            center_x = 960

            rgb_target = get_closest_target(rgb_objs, center_x, SEARCH_DIRECTION)
            ir_target = get_closest_target(ir_objs, center_x, SEARCH_DIRECTION)

            if rgb_target and ir_target:
                tid_rgb, bx_rgb, by_rgb = rgb_target
                tid_ir, bx_ir, by_ir = ir_target
                matched_rgb.append((bx_rgb, by_rgb))
                matched_ir.append((bx_ir, by_ir))

                log_f.write(f"Matched Point P{point_idx}:\n")
                log_f.write(f"  Frame: {frame_id}\n")
                log_f.write(f"  RGB: id={tid_rgb}, (x={bx_rgb:.2f}, y={by_rgb:.2f})\n")
                log_f.write(f"  IR : id={tid_ir}, (x={bx_ir:.2f}, y={by_ir:.2f})\n\n")
                point_idx += 1

    if len(matched_rgb) < 3:
        print(f"❌ 匹配点不足（{len(matched_rgb)}），无法生成单应性矩阵。")
        return

    rgb_pts = np.array(matched_rgb, dtype=np.float32).reshape(-1, 2)
    ir_pts = np.array(matched_ir, dtype=np.float32).reshape(-1, 2)

    H, mask = cv2.findHomography(ir_pts, rgb_pts, method=cv2.RANSAC)
    if H is not None:
        np.save(save_path, H)
        print(f"✅ 单应性矩阵保存成功: {save_path}")
        print(f"共使用匹配点: {len(rgb_pts)}")
    else:
        print("❌ 单应性矩阵计算失败。")


if __name__ == '__main__':
    main()
