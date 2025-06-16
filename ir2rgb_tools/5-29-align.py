
#     main()
import cv2
import numpy as np
import os
import json
from tqdm import tqdm

# ==== 路径配置    ====
# rgb_gt_path = r'/Users/lisushang/Downloads/JieRui2024/datasets/rgb_0275_gt.txt'
# ir_gt_path = r'/Users/lisushang/Downloads/JieRui2024/datasets/inf_0275_gt.txt'
# mask_flag_path = r'/Users/lisushang/Downloads/JieRui2024/datasets/mask_0275gt.txt'
#
# mask_info_path = r'/Users/lisushang/Downloads/JieRui2024/datasets/mask_info.txt'
#
# save_path = '0275_affine_matrix.npy'
#
# rgb_img_template = r'/Users/lisushang/Downloads/jierui24_final_RGB/train/0275/image/{:06d}.jpg'
# ir_img_template  = r'/Users/lisushang/Downloads/jierui24_final_INF/train/0275/image/{:06d}.jpg'

# ==== 可调参数 ====
AREA_THRESH = 40000
NUM_FRAMES_LIMIT = 1000
FRAME_START = 1
VISUALIZE = False
SHOW_IMAGE = False

# ==== 获取最大帧号 ====
def get_max_frame(filepath):
    max_frame = 0
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            try:
                frame = int(float(parts[0]))
                if frame > max_frame:
                    max_frame = frame
            except:
                continue
    return max_frame

# ==== 载入遮挡帧标记 ====
def load_mask_flags(mask_path):
    masked_frames = set()
    with open(mask_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if int(parts[0]) == 1:
                frame_id = int(parts[1])
                masked_frames.add(frame_id)
    return masked_frames
# ==== 载入遮挡帧 ====
def load_mask_info(mask_path,seq_id):
    with open(mask_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data[str(seq_id)]
def get_masked_frames(mask_path, seq_id):
    with open(mask_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    masked_frames = [int(frame_id) for frame_id, info in data[seq_id].items() if info['masked']]
    return masked_frames


# ==== 加载每帧的目标框 ====
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

# ==== 找到离图像中心最近的小目标 ====
# def find_nearest_center_obj(gt_objs, image_center):
#     min_dist = float('inf')
#     best_pt = None
#     for (x, y, w, h) in gt_objs.values():
#         if w * h > AREA_THRESH:
#             continue
#         cx = x + w / 2
#         cy = y + h
#         dist = np.hypot(cx - image_center[0], cy - image_center[1])
#         if dist < min_dist:
#             min_dist = dist
#             best_pt = (cx, cy)
#     return best_pt

def find_nearest_center_obj(gt_objs, image_center):
    min_dist = float('inf')
    best_pt = None
    for (x, y, w, h) in gt_objs.values():
        if w * h > AREA_THRESH:
            continue
        cx = x + w / 2
        cy = y + h/2
        dist = np.hypot(cx - image_center[0], cy - image_center[1])
        if dist < min_dist:
            min_dist = dist
            best_pt = (x, cy)
    return best_pt


# ==== 主函数 ====
def process_by_images(rgb_gt_path,ir_gt_path,mask_info_path,rgb_img_template,ir_img_template,save_path,seq_id=None):
    max_frame = get_max_frame(rgb_gt_path)
    rgb_gt = load_gt_dict_per_frame(rgb_gt_path)
    ir_gt = load_gt_dict_per_frame(ir_gt_path)
    # masked_frames = load_mask_flags(mask_flag_path)
    masked_frames = get_masked_frames(mask_info_path,seq_id)

    matched_rgb, matched_ir = [], []
    frames_checked = 0
    current_frame = FRAME_START

    while current_frame <= max_frame and frames_checked < NUM_FRAMES_LIMIT:
        if current_frame in masked_frames:
            current_frame += 1
            continue

        rgb_objs = rgb_gt.get(current_frame, {})
        ir_objs = ir_gt.get(current_frame, {})

        rgb_img_path = rgb_img_template.format(current_frame)
        ir_img_path = ir_img_template.format(current_frame)
        rgb_img = cv2.imread(rgb_img_path)
        ir_img = cv2.imread(ir_img_path)
        if rgb_img is None or ir_img is None:
            current_frame += 1
            continue

        rgb_center = (rgb_img.shape[1] // 2, rgb_img.shape[0] // 2)
        ir_center = (ir_img.shape[1] // 2, ir_img.shape[0] // 2)

        rgb_pt = find_nearest_center_obj(rgb_objs, rgb_center)
        ir_pt = find_nearest_center_obj(ir_objs, ir_center)

        if rgb_pt and ir_pt:
            matched_rgb.append(rgb_pt)
            matched_ir.append(ir_pt)
            frames_checked += 1

        current_frame += 1

    if len(matched_rgb) < 3:
        print(f"⚠️ 匹配点不足（仅 {len(matched_rgb)} 对），无法估计单应性矩阵。")
        return

    rgb_pts = np.array(matched_rgb, dtype=np.float32)
    ir_pts = np.array(matched_ir, dtype=np.float32)
    H, mask = cv2.findHomography(ir_pts, rgb_pts, method=cv2.RANSAC)

    if H is not None:
        np.save(save_path, H)
        print(f"✅ 单应性矩阵保存成功：{save_path}")
        print(f"共使用匹配点数：{len(rgb_pts)}")
    else:
        print("❌ 单应性矩阵估计失败。")

def process_by_seq(dataset_dir,base_dir,type='dataset'):
    seq_list = os.listdir(os.path.join(dataset_dir, 'train'))
    for seq_id in tqdm(seq_list):
        if seq_id == '.DS_Store':
            continue
        if type=='dataset':
            rgb_gt_path = r'/Users/lisushang/Downloads/jierui24_final_RGB/train/{}/gt/gt_mask.txt'.format(seq_id)
            ir_gt_path = r'/Users/lisushang/Downloads/jierui24_final_INF/train/{}/gt/gt.txt'.format(seq_id)
        else:
            rgb_gt_path = r'{}/{}.txt'.format('/Users/lisushang/Downloads/JieRui2024/jierui_results/juesai/track_outputs_RGB', seq_id)
            ir_gt_path = r'{}/{}.txt'.format('/Users/lisushang/Downloads/JieRui2024/jierui_results/juesai/track_outputs_INF', seq_id)
        mask_info_path = r'{}/mask_info.txt'.format(base_dir)
        save_path = r'{}/{}_affine_matrix.npy'.format(base_dir,seq_id)

        rgb_img_template = os.path.join(dataset_dir,'train',seq_id,r'image/{:06d}.jpg')
        ir_img_template = os.path.join(dataset_dir[:-4]+"INF/",'train',seq_id,r'image/{:06d}.jpg')
        process_by_images(rgb_gt_path,ir_gt_path,mask_info_path,rgb_img_template,ir_img_template,save_path,seq_id=seq_id)
        pass

if __name__ == '__main__':
    dataset_dir = r'/Users/lisushang/Downloads/jierui24_final_RGB/'
    base_dir = r'/Users/lisushang/Downloads/JieRui2024/datasets/'
    os.makedirs(base_dir,exist_ok=True)
    process_by_seq(dataset_dir, base_dir,type='dataset')

