import os
import numpy as np
import pandas as pd
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.byte_tracker import STrack
from tqdm import tqdm


# ===== ByteTrack 配置参数（可根据需要调整） =====
class TrackerArgs:
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    frame_rate = 30
    mot20 = False

args = TrackerArgs()
def det2gt_byte_seq(det_root,dataset_root):

    for seq_file in seq_list:
        seq_id = seq_file[:4]
        det_path = os.path.join(det_root, seq_file)
        save_path = os.path.join(dataset_root, seq_id, 'gt', 'gt_byte.txt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 读取 det 文件
        det_data = pd.read_csv(det_path, header=None).values
        tracker = BYTETracker(args, frame_rate=args.frame_rate)

        results = []
        frame_id = 0
        for frame_id in range(1, int(det_data[:, 0].max()) + 1):
            # 当前帧所有检测
            frame_dets = det_data[det_data[:, 0] == frame_id]
            if frame_dets.shape[0] == 0:
                tracker.update(np.empty((0, 5)), frame_id)
                continue

            bboxes = frame_dets[:, 1:6]  # x, y, w, h, score
            online_targets = tracker.update(bboxes, frame_id)

            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                x, y, w, h = tlwh
                results.append([frame_id, tid, x, y, w, h, 1, -1, -1, -1])

        # 写入 GT 格式
        results = np.array(results)
        np.savetxt(save_path, results, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d')
        print(f"Saved: {save_path}")

if __name__ == '__main__':
    # ===== 路径配置 =====
    det_root = '/Users/lisushang/PycharmProjects/JieRui2024/jierui_final_tools/gt2det2gt_byte/det'  # 你的 det 文件夹
    dataset_root = '/Users/lisushang/Downloads/jierui24_final_RGB/train'  # 原始数据集根目录，包含各序列子目录
    seq_list = sorted([f for f in os.listdir(det_root) if f.endswith('_det.txt')])
    det2gt_byte_seq(det_root,dataset_root)
