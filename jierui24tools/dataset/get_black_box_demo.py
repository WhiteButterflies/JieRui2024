import cv2
import numpy as np


def detect_black_boxes(image_path, area_thresh=500, black_thresh=30):
    # 读取图像（确保3通道）
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image not found or invalid path")

    # 方法1：直接在BGR空间检测纯黑（更严格）
    # black_mask = np.all(img < [black_thresh] * 3, axis=2).astype(np.uint8) * 255

    # 方法2（备选）：灰度图+阈值（适用于灰度场景）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, black_mask = cv2.threshold(gray, black_thresh, 255, cv2.THRESH_BINARY_INV)

    # 去噪（开运算）
    # kernel = np.ones((3, 3), np.uint8)
    # black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    vis_img = img.copy()
    for cnt in contours:
        # 使用最小外接矩形（更精确）
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(int)
        area = cv2.contourArea(cnt)

        if area > area_thresh:
            boxes.append(box)
            cv2.drawContours(vis_img, [box], 0, (0, 0, 255), 2)  # 绘制红色旋转矩形

    return boxes, vis_img

if __name__ == '__main__':

    # 使用示例
    image_path = r'/Users/lisushang/Downloads/jierui24_final_RGB/train/0061/image/000003.jpg'
    black_boxes, vis_img = detect_black_boxes(image_path, area_thresh=500, black_thresh=1)
    cv2.imshow("Result", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
