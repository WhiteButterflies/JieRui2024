import os
import pandas as pd

def generate_gt2det_seq(dataset_root,output_det_folder):

    os.makedirs(output_det_folder, exist_ok=True)

    seq_dirs = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))])

    for seq in seq_dirs:
        gt_path = os.path.join(dataset_root, seq, 'gt', 'gt.txt')
        if not os.path.exists(gt_path):
            continue

        columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
        df = pd.read_csv(gt_path, header=None, names=columns)

        # 模拟检测框，仅保留 ByteTrack 所需字段，并设置 conf=1.0
        df_det = df[['frame', 'x', 'y', 'w', 'h']].copy()
        df_det['score'] = 1.0  # 假设置信度为1.0，可随机也可以按 visibility 映射

        # 保存为 ByteTrack 格式：frame,x,y,w,h,score
        det_output_path = os.path.join(output_det_folder, f"{seq}_det.txt")
        df_det.to_csv(det_output_path, index=False, header=False)
        print(f"Saved: {det_output_path}")
if __name__ == '__main__':
    dataset_root = '/Users/lisushang/Downloads/jierui24_final_RGB/train'  # 替换为实际路径
    output_det_folder = '/Users/lisushang/PycharmProjects/JieRui2024/jierui_final_tools/gt2det/det'
    generate_gt2det_seq(dataset_root,output_det_folder)