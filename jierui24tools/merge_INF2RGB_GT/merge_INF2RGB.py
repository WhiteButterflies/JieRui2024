import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

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

def merge_ir_rgb_sequence(seq_id, rgb_root, ir_root, matrix_dir, id_offset=10000):
    rgb_gt_path = os.path.join(rgb_root, seq_id, "gt", "gt_mask.txt")
    ir_gt_path = os.path.join(ir_root, seq_id, "gt", "gt.txt")
    matrix_path = os.path.join(matrix_dir, f"{seq_id}_affine_matrix.npy")

    # 输出路径
    output_path = os.path.join(rgb_root, seq_id, "gt", "IR_RGB.txt")

    # 加载数据
    columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
    df_rgb = pd.read_csv(rgb_gt_path, header=None, names=columns)
    df_ir = pd.read_csv(ir_gt_path, header=None, names=columns)
    M = np.load(matrix_path)
    # 避免 ID 冲突
    df_ir['id'] += id_offset

    # 应用仿射变换
    df_ir[['x', 'y', 'w', 'h']] = df_ir.apply(lambda row: transform_bbox(row, M), axis=1)

    # 合并
    df_merged = pd.concat([df_rgb, df_ir], ignore_index=True).sort_values(by=['frame', 'id'])

    # 保存
    df_merged.to_csv(output_path, header=False, index=False)
    print(f"[✓] 合并完成: {seq_id} -> {output_path}")

def batch_merge_all_sequences(rgb_root, ir_root, matrix_dir):
    # seq_list = sorted(os.listdir(rgb_root))
    seq_list = ['0061']
    for seq_id in tqdm(seq_list, desc="处理所有序列"):
        try:
            merge_ir_rgb_sequence(seq_id, rgb_root, ir_root, matrix_dir)
        except Exception as e:
            print(f"[X] 序列 {seq_id} 处理失败: {e}")

if __name__ == '__main__':
    batch_merge_all_sequences(rgb_root=r"/Users/lisushang/Downloads/jierui24_final_RGB/train/",ir_root='/Users/lisushang/Downloads/jierui24_final_INF/train/',matrix_dir='/Users/lisushang/PycharmProjects/JieRui2024/jierui24tools/merge_INF2RGB_GT/best_affine')
    # batch_merge_all_sequences(rgb_root=r"/Users/lisushang/Downloads/jierui24_final_RGB/train/",ir_root='/Users/lisushang/Downloads/jierui24_final_INF/train/',matrix_dir='/Users/lisushang/PycharmProjects/JieRui2024/jierui24tools/judge_inf_rgb_area/affine_demo')