import os
import cv2
import pandas as pd

def draw_boxes(img, boxes):
    for _, row in boxes.iterrows():
        x, y, w, h = map(int, [row['x'], row['y'], row['w'], row['h']])
        tid = int(row['id'])
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, str(tid), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main():
    args = {
        'image_dir': r"D:\5-16data\jierui24_final_INF\train\0247\image",
        'mot_txt': r"D:\5-16data\jierui24_final_INF\train\0247\gt\gt.txt"
    }

    # 读取 MOT 格式的txt文件
    df = pd.read_csv(args['mot_txt'], header=None)
    df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'cls', 'vis']
    df['frame'] = df['frame'].astype(int)

    # 获取所有帧图像
    image_files = sorted(
        [f for f in os.listdir(args['image_dir']) if f.endswith(('.jpg', '.png'))]
    )

    for image_name in image_files:
        frame_id = int(os.path.splitext(image_name)[0])  # e.g., "000164" -> 164
        img_path = os.path.join(args['image_dir'], image_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] 图像无法读取：{img_path}")
            continue

        # 提取该帧对应的bbox
        boxes = df[df['frame'] == frame_id]
        draw_boxes(img, boxes)

        cv2.imshow("Image Viewer", img)
        key = cv2.waitKey(0)
        if key == 27:
            print("👋 用户按下 ESC，退出播放。")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# import cv2
# import os
#
# # 配置路径
# image_path = r"D:\5-16data\jierui24_final_RGB\train\0061\image\000164.jpg"
# mot_mask_path = r"D:\JieRui2024\datasets\mask_gt.txt"
# target_frame = 164  # MOT中帧号通常是整数（从1开始）
#
#
# def load_mot_mask_annotations(mask_file, frame_id):
#     masks = []
#     with open(mask_file, 'r') as f:
#         for line in f:
#             parts = line.strip().split(',')
#             if int(parts[0]) == frame_id:
#                 x, y, w, h = map(float, parts[2:6])
#                 masks.append((int(x), int(y), int(w), int(h)))
#     return masks
#
#
# def draw_masks_on_image(img_path, masks):
#     img = cv2.imread(img_path)
#     if img is None:
#         raise FileNotFoundError(f"图像未找到: {img_path}")
#
#     for (x, y, w, h) in masks:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色框
#     return img
#
#
# if __name__ == "__main__":
#     mask_boxes = load_mot_mask_annotations(mot_mask_path, target_frame)
#     image_with_masks = draw_masks_on_image(image_path, mask_boxes)
#
#     # 显示图像
#     cv2.imshow(f"Frame {target_frame} with MOT Masks", image_with_masks)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
