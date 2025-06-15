import os
import sys

import cv2
import pandas as pd

def draw_boxes(img, boxes, color, label_prefix=""):
    for _, row in boxes.iterrows():
        x, y, w, h = map(int, row[['x', 'y', 'w', 'h']])
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        if label_prefix:
            cv2.putText(img, f"{label_prefix}{int(row['id'])}", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def visualize_sequence(seq_id, rgb_root,type='dataset'):
    img_dir = os.path.join(rgb_root, seq_id, "image")

    if type =='dataset':
        original_gt = pd.read_csv(os.path.join(rgb_root, seq_id, "gt", "gt.txt"), header=None)
        merged_gt = pd.read_csv(os.path.join(rgb_root, seq_id, "gt", "IR_RGB.txt"), header=None)
    else:
        results_rgb_dir = r'/Users/lisushang/Downloads/JieRui2024/jierui_results/juesai/track_outputs_RGB'
        original_gt = pd.read_csv(os.path.join(results_rgb_dir,"{}.txt").format(seq_id), header=None)
        results_merge_dir = r'/Users/lisushang/Downloads/JieRui2024/datasets/results'
        merged_gt = pd.read_csv(os.path.join(results_merge_dir,"{}.txt").format(seq_id), header=None)
    columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
    original_gt.columns = merged_gt.columns = columns

    for frame_id in sorted(original_gt['frame'].unique()):
        img_path = os.path.join(img_dir, f"{int(frame_id):06d}.jpg")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)

        rgb_boxes = original_gt[original_gt['frame'] == frame_id]
        merged_boxes = merged_gt[merged_gt['frame'] == frame_id]

        ir_boxes = merged_boxes[~merged_boxes['id'].isin(rgb_boxes['id'])]

        draw_boxes(img, rgb_boxes, (0, 255, 0), label_prefix="RGB_")  # 原GT为绿色
        draw_boxes(img, ir_boxes, (0, 0, 255), label_prefix="IR_")    # 补充的IR为红色

        cv2.putText(img, f"Frame {frame_id}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("GT Merge Visualization", img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
if __name__ == '__main__':
    # visualize_sequence(seq_id='0061', rgb_root=r'D:\5-16data\jierui24_final_RGB\train')
    visualize_sequence(seq_id='0252', rgb_root=r'/Users/lisushang/Downloads/jierui24_final_RGB/train',type='pred')
