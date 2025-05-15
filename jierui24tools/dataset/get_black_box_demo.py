import cv2
import numpy as np


def detect_black_boxes(img, area_thresh=500):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > area_thresh:
            boxes.append((x, y, w, h))
    return boxes


if __name__ == '__main__':

    # 使用示例
    image_path = r'/Users/lisushang/Downloads/jierui24_final_RGB/train/0061/image/000031.jpg'
    black_boxes, vis_img = detect_black_boxes(image_path, area_thresh=500, black_thresh=1)
    cv2.imshow("Result", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
