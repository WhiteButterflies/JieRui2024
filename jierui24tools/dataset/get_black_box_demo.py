import cv2
import numpy as np
import os

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

def process_images_and_save_txt(image_dir, output_txt_path):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    with open(output_txt_path, 'w') as f:
        for frame_id, image_name in enumerate(image_files, start=1):
            img_path = os.path.join(image_dir, image_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            boxes = detect_black_boxes(img)
            for box_id, (x, y, w, h) in enumerate(boxes):
                line = f"{frame_id},{box_id},{x},{y},{w},{h},1,-1,-1\n"
                f.write(line)

if __name__ == '__main__':
    image_dir = r'D:\5-16data\jierui24_final_RGB\train\0252\image'
    output_txt_path = r'D:\JieRui2024\datasets\mask_0252gt.txt'
    process_images_and_save_txt(image_dir, output_txt_path)
