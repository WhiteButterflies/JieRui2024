import cv2
import numpy as np
import os
import json
from tqdm import tqdm

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

def process_images_and_return_dict(image_dir, seq_id):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    seq_dict = {}
    for frame_id, image_name in enumerate(image_files, start=1):
        frame_id_dict = {}
        img_path = os.path.join(image_dir, image_name)
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        if img is None:
            continue
        boxes = detect_black_boxes(img)
        if len(boxes)==0:
            frame_id_dict['masked']=False
            frame_id_dict['masked_area']=None
        else:
            frame_id_dict['masked']=True
            frame_id_dict['masked_area']=[]
            for box_id, (x, y, w, h) in enumerate(boxes):
                frame_id_dict['masked_area'].append((x, y, w, h))
        seq_dict[str(frame_id)] = frame_id_dict
    return seq_dict


def process_seq(dataset_dir,res_dir):
    os.makedirs(res_dir,exist_ok=True)
    output_txt_path = os.path.join(res_dir, 'mask_info.txt')
    result_dict={}
    seq_list = os.listdir(os.path.join(dataset_dir,'train'))
    for seq_id in tqdm(seq_list):
        if seq_id =='.DS_Store':
            continue
        result_dict[str(seq_id)] = process_images_and_return_dict(os.path.join(dataset_dir,'train',seq_id,'image'),seq_id)

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)




if __name__ == '__main__':
    dataset_dir = r'/Users/lisushang/Downloads/jierui24_final_RGB/'
    base_dir = r'/Users/lisushang/Downloads/JieRui2024/datasets/'
    os.makedirs(base_dir,exist_ok=True)
    process_seq(dataset_dir, base_dir)
