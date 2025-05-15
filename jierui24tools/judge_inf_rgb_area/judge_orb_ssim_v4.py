import cv2
import pandas as pd
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import normalized_mutual_information
from tqdm import tqdm
import os

from jierui24tools.judge_inf_rgb_area.judge_cv_ssim_v2 import best_frame


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


def compute_affine_matrix_orb(img_ir, img_rgb):
    # Step 1: 灰度预处理
    gray_ir = preprocess_image(img_ir)
    gray_rgb = preprocess_image(img_rgb)

    # Step 2: 提取 ORB 特征
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(gray_ir, None)
    kp2, des2 = orb.detectAndCompute(gray_rgb, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None  # 无法估计仿射变换

    # Step 3: BFMatcher + 过滤
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 4:
        return None  # 仿射变换至少需要 3 对匹配点

    # Step 4: 提取匹配点
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Step 5: 估计仿射矩阵
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3)
    return M


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


def preprocess_for_edges(img, blur_ksize=3, morph_ksize=1):
    gray = preprocess_image(img)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), sigmaX=1.5)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize, morph_ksize))
    clean_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    return clean_edges


def compare_contours_similarity(img1, img2, mask):
    # 预处理（灰度 + 直方图均衡）
    # gray1 =
    # gray2 =

    # 边缘检测
    edges1 = preprocess_for_edges(img1)
    edges2 = preprocess_for_edges(img2)
    # mask
    x, y, w, h = cv2.boundingRect(mask)
    img1_masked = edges1[y:y + h, x:x + w]
    img2_masked = edges2[y:y + h, x:x + w]
    combined_mask = np.vstack((img1_masked, img2_masked))
    cv2.imshow("combined_mask", combined_mask)

    # 查找轮廓
    contours1, _ = cv2.findContours(img1_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(img2_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours1 or not contours2:
        return 1.0  # 最大差异

    # 选取最大轮廓（面积最大）
    cnt1 = max(contours1, key=cv2.contourArea)
    cnt2 = max(contours2, key=cv2.contourArea)

    # 计算轮廓相似度（越小越相似）
    score = cv2.matchShapes(cnt1, cnt2, method=cv2.CONTOURS_MATCH_I1, parameter=0.0)
    return score


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

    # 方法3
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
    return img[y:y + h, x:x + w]


def generate_matrix_Seq(RGB_IMG_DIR,IR_IMG_DIR,RGB_GT_PATH,IR_GT_PATH):
    FRAME_START, FRAME_END = 1, len(os.listdir(RGB_IMG_DIR))+1  # 改成你的帧范围
    # ============================


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

        #type 1 bbox
        M = compute_affine_align_matrix(ir_box,rgb_box)
        # M = np.load("affine_matrix.npy")
        # if frame_id ==1:
        #     np.save("affine_matrix.npy", M)
        #     exit(0)
        #type 2 ORB
        # M = compute_affine_matrix_orb(img_ir, img_rgb)
        # if M is None:
        #     continue  # 匹配失败，跳过当前帧

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
        ssim_score = compute_masked_ssim(ir_patch, rgb_patch, mask)
        # psnr_score = compute_masked_psnr(ir_patch, rgb_patch, mask)
        # ssim_score = feature_match_orb(ir_patch, rgb_patch,mask)
        # ssim_score = compare_contours_similarity(ir_patch, rgb_patch,mask)

        # 显示合并图像（RGB + IR）和帧号
        combined = np.vstack((aligned_ir, img_rgb))
        text = f"Frame: {frame_id}, SSIM: {ssim_score:.2f}"
        cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Aligned IR vs RGB", combined)

        if ssim_score > best_ssim:
            best_ssim = ssim_score
            best_frame = frame_id
            best_M = M.copy()

    cv2.destroyAllWindows()
    # ========== OUTPUT ==========
    print(f"\nBest SSIM frame: {best_frame}")
    print("Best Affine Matrix (IR → RGB):\n", best_M)
    return best_frame, best_M

def batch_generate_martix_all_sequences(rgb_root, ir_root, matrix_dir):
    seq_list = sorted(os.listdir(rgb_root))
    seq_list = ['0061']
    os.makedirs(matrix_dir, exist_ok=True)
    for seq_id in tqdm(seq_list, desc="处理所有序列"):
        RGB_IMG_DIR = os.path.join(rgb_root, seq_id,'image')
        IR_IMG_DIR = os.path.join(ir_root, seq_id,'image')
        RGB_GT_PATH = os.path.join(rgb_root, seq_id,'gt','gt_mask.txt')
        IR_GT_PATH = os.path.join(ir_root, seq_id,'gt','gt.txt')
        try:
            best_frame,best_m=generate_matrix_Seq(RGB_IMG_DIR,IR_IMG_DIR,RGB_GT_PATH,IR_GT_PATH)
            np.save(os.path.join(matrix_dir,seq_id+"_affine_matrix.npy"), best_m)
        except Exception as e:
            print(f"[X] 序列 {seq_id} 处理失败: {e}")
if __name__ == '__main__':
    # ========== CONFIG ==========
    rgb_root = r"/Users/lisushang/Downloads/jierui24_final_RGB/train/"
    ir_root = r'/Users/lisushang/Downloads/jierui24_final_INF/train/'
    matrix_dir = r'/Users/lisushang/PycharmProjects/JieRui2024/jierui24tools/judge_inf_rgb_area/affine_demo'
    batch_generate_martix_all_sequences(rgb_root, ir_root,matrix_dir)
