import cv2
import pandas as pd
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import normalized_mutual_information

import os

# ========== CONFIG ==========
RGB_IMG_DIR = r"C:\Users\liuji\Downloads/jierui24_final/0061/visible/image"
IR_IMG_DIR = r"C:\Users\liuji\Downloads/jierui24_final/0061/infrared/image"
RGB_GT_PATH = r"C:\Users\liuji\Downloads/jierui24_final_RGB/train/0061/gt/gt_mask.txt"
IR_GT_PATH = r"C:\Users\liuji\Downloads/jierui24_final_INF/train/0061/gt/gt.txt"

RGB_IMG_DIR = r"/Users/lisushang/Downloads/jierui24_final_RGB/train/0061/image"
IR_IMG_DIR = r"/Users/lisushang/Downloads/jierui24_final_INF/train/0061/image"
RGB_GT_PATH = r"/Users/lisushang/Downloads/jierui24_final_RGB/train/0061/gt/gt_mask.txt"
IR_GT_PATH = r"/Users/lisushang/Downloads/jierui24_final_INF/train/0061/gt/gt.txt"
FRAME_START, FRAME_END = 1, 119  # 改成你的帧范围
# ============================

# ========== UTILITIES ==========
def preprocess_image(img):
    """归一化+直方图均衡化提升对比度"""
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    if img.ndim == 3:  # 彩色转灰度
        img_gray = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_norm
    return cv2.equalizeHist(img_gray)  # 直方图均衡化

def compute_motion_vector(df, frame, win=3):
    sub = df[(df['frame'] >= frame - win // 2) & (df['frame'] <= frame + win // 2)]
    vectors = {}
    for obj_id, g in sub.groupby('id'):
        g = g.sort_values('frame')
        if len(g) >= 2:
            center = g[['x', 'y']].values + g[['w', 'h']].values / 2
            h = g['h'].values
            motion = (center[1:] - center[:-1]) / h[:-1, np.newaxis]
            vectors[obj_id] = np.mean(motion, axis=0)
    return vectors

def compute_affine_align_matrix(ir_box, rgb_box):
    x_ir, y_ir, w_ir, h_ir = ir_box
    x_rgb, y_rgb, w_rgb, h_rgb = rgb_box

    scale_w = w_rgb / w_ir
    scale_h = h_rgb / h_ir

    cx_ir = x_ir + w_ir / 2
    cy_ir = y_ir + h_ir / 2
    cx_rgb = x_rgb + w_rgb / 2
    cy_rgb = y_rgb + h_rgb / 2

    tx = cx_rgb - cx_ir * scale_w
    ty = cy_rgb - cy_ir * scale_h

    return np.array([[scale_w, 0, tx], [0, scale_h, ty]], dtype=np.float32)

def get_nonzero_mask(aligned_ir):
    # 将图像转为灰度并检测非零像素
    gray = cv2.cvtColor(aligned_ir, cv2.COLOR_BGR2GRAY)
    mask = (gray > 0).astype(np.uint8) * 255  # 非零区域为255，其余为0
    return mask

def compute_mi(img1, img2, mask):
    x, y, w, h = cv2.boundingRect(mask)
    img1_masked = img1[y:y + h, x:x + w]
    img2_masked = img2[y:y + h, x:x + w]
    combined_mask = np.vstack((img1_masked, img2_masked))
    cv2.imshow("combined_mask", combined_mask)

    return normalized_mutual_information(img1_masked, img2_masked)
def feature_match_orb(img1, img2, mask):
    # 1. 使用掩码裁剪有效区域
    x, y, w, h = cv2.boundingRect(mask)
    img1_masked = img1[y:y + h, x:x + w]
    img2_masked = img2[y:y + h, x:x + w]

    # 2. 初始化 ORB 检测器
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1_masked, None)
    kp2, des2 = orb.detectAndCompute(img2_masked, None)

    # 3. 检查关键点和描述符有效性
    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return 0.0  # 无关键点或描述符时返回 0

    # 4. 确保描述符维度一致（ORB 默认应为 32，但需验证）
    if des1.shape[1] != des2.shape[1]:
        return 0.0  # 描述符维度不一致时直接返回 0

    # 5. 匹配描述符
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        matches = bf.match(des1, des2)
        match_ratio = len(matches) / max(len(kp1), len(kp2))
    except cv2.error as e:
        print(f"匹配错误: {e}")
        return 0.0

    return match_ratio




def compute_masked_ssim(img1, img2, mask):
    # 转为灰度 直方图
    img1_gray = preprocess_image(img1)
    img2_gray = preprocess_image(img2)



    # 检查掩码区域是否足够大（至少 7x7 像素）
    if np.sum(mask) < 7 * 7:
        return 0.0  # 无效区域返回默认值

    # 提取掩码区域 type1
    # img1_masked = img1_gray[mask.astype(bool)]
    # img2_masked = img2_gray[mask.astype(bool)]

    # 方法2：用 bitwise_and 保持形状
    # img1_masked = cv2.bitwise_and(img1_gray, img1_gray, mask=mask)
    # img2_masked = cv2.bitwise_and(img2_gray, img2_gray, mask=mask)

    #方法3
    x, y, w, h = cv2.boundingRect(mask)
    img1_masked = img1_gray[y:y + h, x:x + w]
    img2_masked = img2_gray[y:y + h, x:x + w]

    combined_mask = np.vstack((img1_masked, img2_masked))
    cv2.imshow("combined_mask", combined_mask)

    # 计算 SSIM（单通道，无需 channel_axis）
    return ssim(
        img1_masked,
        img2_masked,
        win_size=7,  # 默认 7x7 窗口
        data_range=255,  # 8-bit 图像范围 0-255
    )


def compute_masked_psnr(img1, img2, mask):
    mask = mask.astype(bool)
    mse = np.mean((img1[mask] - img2[mask]) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def extract_patch(img, box):
    x, y, w, h = map(int, box)
    return img[y:y+h, x:x+w]

# ========== MAIN ==========

columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
inf_gt = pd.read_csv(IR_GT_PATH, header=None, names=columns)
rgb_gt = pd.read_csv(RGB_GT_PATH, header=None, names=columns)

best_ssim = -1
best_frame = -1
best_M = None

for frame_id in range(FRAME_START, FRAME_END):
    img_ir_path = os.path.join(IR_IMG_DIR, f"{frame_id:06d}.jpg")
    img_rgb_path = os.path.join(RGB_IMG_DIR, f"{frame_id:06d}.jpg")
    img_ir = cv2.imread(img_ir_path)
    img_rgb = cv2.imread(img_rgb_path)
    if img_ir is None or img_rgb is None:
        continue

    inf_vecs = compute_motion_vector(inf_gt, frame_id)
    rgb_vecs = compute_motion_vector(rgb_gt, frame_id)

    best_pair = None
    best_sim = -1
    for iid, ivec in inf_vecs.items():
        for rid, rvec in rgb_vecs.items():
            sim = cosine_similarity([ivec], [rvec])[0][0]
            if sim > best_sim:
                best_sim = sim
                best_pair = (iid, rid)

    if best_pair is None:
        continue
    if ((inf_gt['frame'] == frame_id) & (inf_gt['id'] == best_pair[0])).sum()==0 or ((rgb_gt['frame'] == frame_id) & (rgb_gt['id'] == best_pair[1])).sum()== 0:
        continue
    ir_box = inf_gt[(inf_gt['frame'] == frame_id) & (inf_gt['id'] == best_pair[0])][['x', 'y', 'w', 'h']].iloc[0].values
    rgb_box = rgb_gt[(rgb_gt['frame'] == frame_id) & (rgb_gt['id'] == best_pair[1])][['x', 'y', 'w', 'h']].iloc[0].values

    M = compute_affine_align_matrix(ir_box, rgb_box)

    '''calc by box'''
        # h_rgb, w_rgb = img_rgb.shape[:2]
        # aligned_ir = cv2.warpAffine(img_ir, M, (w_rgb, h_rgb), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        #
        # ir_patch = extract_patch(aligned_ir, rgb_box)
        # rgb_patch = extract_patch(img_rgb, rgb_box)
    '''ended'''
    '''calc by RGB and INF overlap'''
    # Step 5: warp IR image
    h_rgb, w_rgb = img_rgb.shape[:2]
    aligned_ir = cv2.warpAffine(img_ir, M, (w_rgb, h_rgb), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # 生成掩码并裁剪有效区域
    mask = get_nonzero_mask(aligned_ir)
    ir_patch = cv2.bitwise_and(aligned_ir, aligned_ir, mask=mask)
    rgb_patch = img_rgb.copy()
    '''ended'''
    '''calc psnr ssim'''
    if ir_patch.shape != rgb_patch.shape:
        ir_patch = cv2.resize(ir_patch, (rgb_patch.shape[1], rgb_patch.shape[0]))

    # 计算掩码区域的相似度
    # ssim_score = compute_masked_ssim(ir_patch, rgb_patch, mask)
    # psnr_score = compute_masked_psnr(ir_patch, rgb_patch, mask)
    ssim_score = feature_match_orb(ir_patch, rgb_patch,mask)

    # 显示合并图像（RGB + IR）和帧号
    combined = np.vstack((img_rgb, aligned_ir))
    text = f"Frame: {frame_id}, SSIM: {ssim_score:.2f}"
    cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("RGB vs Aligned IR", combined)

    # 按 'q' 退出，其他键继续
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

    if ssim_score > best_ssim:
        best_ssim = ssim_score
        best_frame = frame_id
        best_M = M.copy()

cv2.destroyAllWindows()
# ========== OUTPUT ==========
print(f"\nBest SSIM frame: {best_frame}")
print("Best Affine Matrix (IR → RGB):\n", best_M)
