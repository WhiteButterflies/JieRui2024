# #
# # import os
# # import cv2
# # import pandas as pd
# #
# # def draw_boxes(img, boxes):
# #     for _, row in boxes.iterrows():
# #         x, y, w, h = map(int, [row['x'], row['y'], row['w'], row['h']])
# #         tid = int(row['id'])
# #         color = (0, 255, 0)
# #         cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
# #         cv2.putText(img, str(tid), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
# #
# # def main():
# #     # 👇 修改为你的路径
# #     args = {
# #         'image_dir': r"D:\5-16data\jierui24_final_RGB\train\0061\image",
# #         'gt_file': r"D:\5-20-data\txt_save\mask_gt.txt",
# #         'output_dir': r"D:\5-20-data"
# #     }
# #
# #     os.makedirs(args['output_dir'], exist_ok=True)
# #     columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
# #     df = pd.read_csv(args['gt_file'], header=None, names=columns)
# #
# #     for frame_id in sorted(df['frame'].unique()):
# #         filename = f"{int(frame_id):06d}.jpg"
# #         img_path = os.path.join(args['image_dir'], filename)
# #         if not os.path.exists(img_path):
# #             print(f"[WARN] Image not found: {img_path}")
# #             continue
# #
# #         img = cv2.imread(img_path)
# #         boxes = df[df['frame'] == frame_id]
# #         draw_boxes(img, boxes)
# #
# #         # 显示图像
# #         cv2.imshow("GT Visualization", img)
# #         key = cv2.waitKey(0)
# #
# #         if key == 27:  # ESC 键
# #             print("👋 用户按下 ESC，退出。")
# #             break
# #
# #         # 保存结果
# #         # out_path = os.path.join(args['output_dir'], filename)
# #         # cv2.imwrite(out_path, img)
# #         # print(f"[OK] 已保存: {out_path}")
# #
# #     cv2.destroyAllWindows()
# #
# # if __name__ == "__main__":
# #     main()
# import os
# import cv2
# import pandas as pd
#
# def draw_boxes(img, boxes):
#     for _, row in boxes.iterrows():
#         x, y, w, h = map(int, [row['x'], row['y'], row['w'], row['h']])
#         tid = int(row['id'])
#         color = (0, 255, 0)
#         cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#         cv2.putText(img, str(tid), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#
# def main():
#     # 👇 修改为你的路径
#     args = {
#         'image_dir': r"D:\5-16data\jierui24_final_RGB\train\0061\image",
#         'gt_file': r"D:\5-20-data\txt_save\mask_gt.txt",
#         'output_dir': r"D:\5-20-data"
#     }
#
#     os.makedirs(args['output_dir'], exist_ok=True)
#     columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
#     df = pd.read_csv(args['gt_file'], header=None, names=columns)
#
#     for frame_id in sorted(df['frame'].unique()):
#         filename = f"{int(frame_id):06d}.jpg"
#         img_path = os.path.join(args['image_dir'], filename)
#         if not os.path.exists(img_path):
#             print(f"[WARN] Image not found: {img_path}")
#             continue
#
#         img = cv2.imread(img_path)
#         if img is None:
#             continue
#
#         original_img = img.copy()  # 保存原图副本
#         boxes = df[df['frame'] == frame_id]
#         draw_boxes(img, boxes)
#
#         # 显示两个窗口
#         cv2.imshow("Original Image", original_img)
#         cv2.imshow("With GT Boxes", img)
#         key = cv2.waitKey(0)
#
#         if key == 27:  # ESC 键退出
#             print("👋 用户按下 ESC，退出。")
#             break
#
#         # 可选保存
#         # out_path_with_box = os.path.join(args['output_dir'], f"{filename}_with_box.jpg")
#         # out_path_original = os.path.join(args['output_dir'], f"{filename}_original.jpg")
#         # cv2.imwrite(out_path_with_box, img)
#         # cv2.imwrite(out_path_original, original_img)
#         # print(f"[OK] 已保存: {out_path_with_box} 和 {out_path_original}")
#
#     cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     main()
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
        'image_dir': r"D:\5-16data\jierui24_final_RGB\train\0275\image",
        'gt_file': r"D:\JieRui2024\datasets\mask_0275gt.txt",
        'output_dir': r"D:\5-20-data"
    }

    os.makedirs(args['output_dir'], exist_ok=True)
    columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
    df = pd.read_csv(args['gt_file'], header=None, names=columns)

    image_files = sorted([f for f in os.listdir(args['image_dir']) if f.endswith(('.jpg', '.png'))])

    for image_name in image_files:
        frame_id = int(os.path.splitext(image_name)[0])
        img_path = os.path.join(args['image_dir'], image_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        original_img = img.copy()
        boxes = df[df['frame'] == frame_id]

        # 仅在该帧有 box 的情况下才画框
        if not boxes.empty:
            draw_boxes(img, boxes)

        # 显示两个窗口
        cv2.imshow("Original Image", original_img)
        cv2.imshow("With GT Boxes", img)
        key = cv2.waitKey(0)
        if key == 27:
            print("👋 用户按下 ESC，退出。")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
