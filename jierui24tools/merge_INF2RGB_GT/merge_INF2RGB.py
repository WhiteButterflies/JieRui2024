import os
import cv2
import numpy as np
import pandas as pd
# from Cython.Includes.cpython.time import result
from tqdm import tqdm
import json

def transform_bbox(row, M):
    x, y, w, h = row['x'], row['y'], row['w'], row['h']
    pts = np.array([
        [x, y],
        [x + w, y],
        [x, y + h],
        [x + w, y + h]
    ], dtype=np.float32)
    pts_trans = cv2.transform(np.array([pts]), M)[0]
    x_new, y_new = pts_trans[:, 0].min(), pts_trans[:, 1].min()
    w_new = pts_trans[:, 0].max() - x_new
    h_new = pts_trans[:, 1].max() - y_new
    return pd.Series([x_new, y_new, w_new, h_new])
def load_mask_info(mask_path,seq_id):
    with open(mask_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[str(seq_id)]

def get_first_masked_area(mask_path, seq_id):
    data = load_mask_info(mask_path, seq_id)
    for frame_id in sorted(data, key=lambda x: int(x)):
        info = data[frame_id]
        if info['masked']:
            return info['masked_area']
    return None  # 如果没有 masked==True 的帧

def merge_ir_rgb_sequence(seq_id, rgb_root, ir_root, matrix_dir, mask_info,type='dataset',id_offset=10000):
    if type =='dataset':
        rgb_gt_path = os.path.join(rgb_root, seq_id, "gt", "gt_mask.txt")
        ir_gt_path = os.path.join(ir_root, seq_id, "gt", "gt.txt")
    else:
        result_rgb_dir = r'/Users/lisushang/Downloads/JieRui2024/jierui_results/juesai/track_outputs_RGB/'
        result_inf_dir = r'/Users/lisushang/Downloads/JieRui2024/jierui_results/juesai/track_outputs_INF/'
        rgb_gt_path = os.path.join(result_rgb_dir,"{}.txt".format(seq_id))
        ir_gt_path = os.path.join(result_inf_dir,"{}.txt".format(seq_id))
    matrix_path = os.path.join(matrix_dir, f"{seq_id}_affine_matrix.npy")
    if type =='dataset':
        output_path = os.path.join(rgb_root, seq_id, "gt", "IR_RGB.txt")
    else:
        result_dir = os.path.join(matrix_dir,"results")
        os.makedirs(result_dir,exist_ok=True)
        output_path = os.path.join(result_dir,"{}.txt".format(seq_id))
    # 检查文件是否存在
    if not all(os.path.exists(p) for p in [rgb_gt_path, ir_gt_path, matrix_path, mask_info]):
        missing = [p for p in [rgb_gt_path, ir_gt_path, matrix_path, mask_info] if not os.path.exists(p)]
        print(f"[!] 跳过 {seq_id}：缺少文件: {', '.join(missing)}")
        return

    # 加载数据
    columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
    df_rgb = pd.read_csv(rgb_gt_path, header=None, names=columns)
    df_ir = pd.read_csv(ir_gt_path, header=None, names=columns)
    H = np.load(matrix_path)


    # 变换IR框并过滤
    transformed_ir = []
    for _, row in df_ir.iterrows():
        frame = row['frame']
        masks = get_first_masked_area(mask_info,seq_id)
        if not masks:
            continue

        # 使用单应性变换中心点
        cx, cy = row['x'] + row['w']/2, row['y'] + row['h']
        pt = np.array([cx, cy, 1.0]).reshape(3,1)
        dst = H @ pt
        dst /= dst[2]
        nx, ny = dst[0,0] - row['w']/2, dst[1,0] - row['h']

        # 检查是否在mask区域内
        if any((nx + row['w'] > mx and nx < mx + mw and
                ny + row['h'] > my and ny < my + mh)
               for mx, my, mw, mh in masks):
            transformed_ir.append([
                int(frame),
                row['id'] + id_offset,
                nx, ny, row['w'], row['h'],
                row['conf'], row['class'], row['vis']
            ])

    # 创建DataFrame并合并
    df_ir_trans = pd.DataFrame(transformed_ir, columns=columns)
    df_merged = pd.concat([df_rgb, df_ir_trans], ignore_index=True).sort_values(by=['frame', 'id'])
    df_merged.to_csv(output_path, header=False, index=False)
    print(f"[✓] 合并完成: {seq_id} -> {output_path} (保留 {len(df_ir_trans)}/{len(df_ir)} IR检测)")

def batch_merge_all_sequences(rgb_root, ir_root, matrix_dir,mask_info,type='dataset'):
    seq_list = sorted([
        s for s in os.listdir(rgb_root)
        if os.path.isdir(os.path.join(rgb_root, s)) and not s.startswith(".")
    ])
    for seq_id in tqdm(seq_list, desc="处理所有序列"):
        try:
            merge_ir_rgb_sequence(seq_id, rgb_root, ir_root, matrix_dir,mask_info,type)
        except Exception as e:
            print(f"[X] 序列 {seq_id} 处理失败: {e}")

if __name__ == '__main__':
    # batch_merge_all_sequences(
    #     rgb_root=r"D:\5-16data\jierui24_final_RGB\train",
    #     ir_root=r"D:\5-16data\jierui24_final_INF\train",
    #     matrix_dir=r"D:\JieRui2024\jierui24tools\merge_INF2RGB_GT\best_affine"
    # )
    batch_merge_all_sequences(
        rgb_root=r"/Users/lisushang/Downloads/jierui24_final_RGB/train/",
        ir_root=r"/Users/lisushang/Downloads/jierui24_final_INF/train/",
        matrix_dir=r"/Users/lisushang/Downloads/JieRui2024/datasets",
        mask_info=r'/Users/lisushang/Downloads/JieRui2024/datasets/mask_info.txt',
        # type='dataset'
        type='pred'
    )
