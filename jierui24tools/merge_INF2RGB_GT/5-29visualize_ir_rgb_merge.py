import os
import cv2
import numpy as np
import pandas as pd

# ====== 配置路径 ======
rgb_gt_path     = r"/Users/lisushang/Downloads/jierui24_final_RGB/train/0188/gt/gt_mask.txt"
ir_gt_path      = r'/Users/lisushang/Downloads/JieRui2024/datasets/inf_0188_gt.txt'
mask_gt_path    = r'/Users/lisushang/Downloads/JieRui2024/datasets/mask_0188gt.txt'
# H_path          = r"/Users/lisushang/Downloads/JieRui2024/ir2rgb_tools/0188_affine_matrix.npy"
H_path          = r"/Users/lisushang/Downloads/JieRui2024/datasets/0188_affine_matrix.npy"
rgb_img_dir     = r"/Users/lisushang/Downloads/jierui24_final_RGB/train/0188/image/"



# ====== 工具函数 ======
def load_mot_gt_by_frame(filepath):
    df = pd.read_csv(filepath, header=None)
    d = {}
    for _, r in df.iterrows():
        f, tid = int(r[0]), int(r[1])
        x, y, w, h = map(float, r[2:6])
        d.setdefault(f, []).append((tid, x, y, w, h))
    return d

def load_mask_regions(filepath):
    df = pd.read_csv(filepath, header=None)
    m = {}
    for _, r in df.iterrows():
        f = int(r[0])
        x, y, w, h = map(float, r[2:6])
        m.setdefault(f, []).append((x, y, w, h))
    return m

def is_inside_mask(x, y, w, h, masks):
    # 判断变换后框是否与任一 mask 矩形有交集
    for mx, my, mw, mh in masks:
        if (x + w > mx and x < mx + mw and
            y + h > my and y < my + mh):
            return True
    return False

def transform_ir_boxes(ir_boxes, H, masks):
    out = []
    for tid, x, y, w, h in ir_boxes:
        pt = np.array([x + w/2, y + h, 1.0]).reshape(3,1)
        dst = H @ pt
        dst /= dst[2]
        cx, cy = dst[0,0], dst[1,0]
        nx, ny = cx - w/2, cy - h
        if is_inside_mask(nx, ny, w, h, masks):
            out.append((tid, nx, ny, w, h))
    return out

def draw_boxes(img, boxes, color, prefix):
    for tid, x, y, w, h in boxes:
        p1 = (int(x), int(y))
        p2 = (int(x+w), int(y+h))
        ctr = (int(x+w/2), int(y+h))
        cv2.rectangle(img, p1, p2, color, 2)
        cv2.putText(img, f"{prefix}{tid}", (ctr[0]+5, ctr[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

# ====== 主流程 ======
def main():
    print("加载数据…")
    rgb_dict  = load_mot_gt_by_frame(rgb_gt_path)
    ir_dict   = load_mot_gt_by_frame(ir_gt_path)
    mask_dict = load_mask_regions(mask_gt_path)
    H         = np.load(H_path)

    frames = sorted(set(mask_dict.keys()) & set(ir_dict.keys()))
    for f in frames:
        img_path = os.path.join(rgb_img_dir, f"{f:06d}.jpg")
        if not os.path.exists(img_path):
            print(f"[跳过] Frame {f:06d} 图像不存在")
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        # 绘制 RGB 框
        rgb_boxes = rgb_dict.get(f, [])
        draw_boxes(img, rgb_boxes, (0,255,0), prefix="RGB")

        # 变换 & 过滤 IR 框
        ir_boxes = ir_dict.get(f, [])
        masks    = mask_dict.get(f, [])
        if ir_boxes and masks:
            ir_t = transform_ir_boxes(ir_boxes, H, masks)
            draw_boxes(img, ir_t, (0,0,255), prefix="IR")

        # 标注帧号并显示
        cv2.putText(img, f"Frame {f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("IR→RGB Fused Viewer", img)
        k = cv2.waitKey(0)
        if k == ord('q'):
            break

    cv2.destroyAllWindows()
    print("可视化结束。")

if __name__ == "__main__":
    main()
