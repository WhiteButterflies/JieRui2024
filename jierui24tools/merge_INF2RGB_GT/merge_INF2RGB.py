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

def merge_ir_rgb_sequence(seq_id, rgb_root, ir_root, matrix_dir, mask_info, type='dataset', id_offset=10000):
    if type == 'dataset':
        rgb_gt_path = os.path.join(rgb_root, seq_id, "gt", "gt_mask.txt")
        ir_gt_path = os.path.join(ir_root, seq_id, "gt", "gt.txt")
    else:
        result_rgb_dir = r'/Users/lisushang/Downloads/JieRui2024/jierui_results/juesai/track_outputs_RGB/'
        result_inf_dir = r'/Users/lisushang/Downloads/JieRui2024/jierui_results/juesai/track_outputs_INF/'
        rgb_gt_path = os.path.join(result_rgb_dir, "{}.txt".format(seq_id))
        ir_gt_path = os.path.join(result_inf_dir, "{}.txt".format(seq_id))

    matrix_path = os.path.join(matrix_dir, f"{seq_id}_affine_matrix.npy")

    if type == 'dataset':
        output_path = os.path.join(rgb_root, seq_id, "gt", "IR_RGB.txt")

        # result_dir = os.path.join(matrix_dir, "results")
        # os.makedirs(result_dir, exist_ok=True)
        # output_path = os.path.join(result_dir, "{}.txt".format(seq_id))
    else:
        result_dir = os.path.join(matrix_dir, "results")
        os.makedirs(result_dir, exist_ok=True)
        output_path = os.path.join(result_dir, "{}.txt".format(seq_id))

    # 检查文件是否存在
    if not all(os.path.exists(p) for p in [rgb_gt_path, ir_gt_path, matrix_path, mask_info]):
        missing = [p for p in [rgb_gt_path, ir_gt_path, matrix_path, mask_info] if not os.path.exists(p)]
        print(f"[!] 跳过 {seq_id}：缺少文件: {', '.join(missing)}")
        return

    # 加载数据
    columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
    dtype_mapping = {
        'frame': 'int32',
        'id': 'int32',
        'x': 'float32',
        'y': 'float32',
        'w': 'float32',
        'h': 'float32',
        'conf': 'float32',
        'class': 'int32',
        'vis': 'float32'
    }
    df_rgb = pd.read_csv(rgb_gt_path, header=None, names=columns, dtype=dtype_mapping)
    df_ir = pd.read_csv(ir_gt_path, header=None, names=columns, dtype=dtype_mapping)
    H = np.load(matrix_path)

    # 加载所有mask信息
    with open(mask_info, 'r', encoding='utf-8') as f:
        all_mask_data = json.load(f)
    seq_mask_data = all_mask_data.get(str(seq_id), {})

    # 按帧处理
    merged_results = []
    all_frames = sorted(set(df_rgb['frame']).union(set(df_ir['frame'])))

    for frame in all_frames:
        # 获取当前帧的mask区域
        frame_mask = seq_mask_data.get(str(frame), {}).get('masked_area', [])
        if frame_mask is not  None:

            # 当前帧的RGB检测
            rgb_frame = df_rgb[df_rgb['frame'] == frame]

            # 当前帧的IR检测
            ir_frame = df_ir[df_ir['frame'] == frame]

            # 变换并过滤IR检测
            transformed_ir = []
            for _, row in ir_frame.iterrows():
                # 使用单应性变换中心点
                cx, cy = row['x'] + row['w']/2, row['y'] + row['h']
                pt = np.array([cx, cy, 1.0]).reshape(3,1)
                dst = H @ pt
                dst /= dst[2]
                nx, ny = dst[0,0] - row['w']/2, dst[1,0] - row['h']

                # 检查是否在mask区域内
                if any((nx + row['w'] > mx and nx < mx + mw and
                        ny + row['h'] > my and ny < my + mh)
                       for mx, my, mw, mh in frame_mask):
                    transformed_ir.append({
                        'frame': frame,
                        'id': row['id'] + id_offset,
                        'x': nx,
                        'y': ny,
                        'w': row['w'],
                        'h': row['h'],
                        'conf': row['conf'],
                        'class': row['class'],
                        'vis': row['vis']
                    })

            # 合并当前帧的结果
            merged_results.extend(rgb_frame.to_dict('records'))
            merged_results.extend(transformed_ir)
        else:
            # 当前帧的RGB检测
            rgb_frame = df_rgb[df_rgb['frame'] == frame]
            merged_results.extend(rgb_frame.to_dict('records'))

    # 创建最终的DataFrame并保存
    df_merged = pd.DataFrame(merged_results, columns=columns).sort_values(by=['frame', 'id'])
    df_merged.to_csv(output_path, header=False, index=False)

    total_ir = len(df_ir)
    kept_ir = len(df_merged[df_merged['id'] >= id_offset])
    print(f"[✓] 合并完成: {seq_id} -> {output_path} (保留 {kept_ir}/{total_ir} IR检测)")

def batch_merge_all_sequences(rgb_root, ir_root, matrix_dir,mask_info,type='dataset'):
    seq_list = sorted([
        s for s in os.listdir(rgb_root)
        if os.path.isdir(os.path.join(rgb_root, s)) and not s.startswith(".")
    ])
    for seq_id in tqdm(seq_list, desc="处理所有序列"):
        # try:
        merge_ir_rgb_sequence(seq_id, rgb_root, ir_root, matrix_dir,mask_info,type)
        # except Exception as e:
        #     print(f"[X] 序列 {seq_id} 处理失败: {e}")

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
