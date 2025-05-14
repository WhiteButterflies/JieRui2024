# Re-run after code state reset
import cv2
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Reload all inputs
inf_gt_path = "/Users/lisushang/Downloads/jierui24_final_INF/train/0061/gt/gt.txt"
rgb_gt_path = "/Users/lisushang/Downloads/jierui24_final_RGB/train/0061/gt/gt_mask.txt"


columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
inf_gt = pd.read_csv(inf_gt_path, header=None, names=columns)
rgb_gt = pd.read_csv(rgb_gt_path, header=None, names=columns)

# Step 1: compute motion vector
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
 # Step 4: compute affine transform
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

    M = np.array([
        [scale_w, 0, tx],
        [0, scale_h, ty]
    ], dtype=np.float32)
    return M

global_M =None
for frame_id in range(1,rgb_gt.shape[0]):
    # img_ir_path = f"/Users/lisushang/Downloads/jierui24_final/0188/infrared/image/{int(frame_id):06d}.jpg"
    # img_rgb_path = f"/Users/lisushang/Downloads/jierui24_final/0188/visible/image/{int(frame_id):06d}.jpg"
    img_ir_path = f"/Users/lisushang/Downloads/jierui24_final_INF/train/0061/image/{int(frame_id):06d}.jpg"
    img_rgb_path = f"/Users/lisushang/Downloads/jierui24_final_RGB/train/0061/image/{int(frame_id):06d}.jpg"
    img_ir = cv2.imread(img_ir_path)
    img_rgb = cv2.imread(img_rgb_path)

    inf_vecs = compute_motion_vector(inf_gt, frame_id)
    rgb_vecs = compute_motion_vector(rgb_gt, frame_id)

    # Step 2: match most similar pair
    best_pair = None
    best_sim = -1
    for iid, ivec in inf_vecs.items():
        for rid, rvec in rgb_vecs.items():
            sim = cosine_similarity([ivec], [rvec])[0][0]
            if sim > best_sim:
                best_sim = sim
                best_pair = (iid, rid)

    # Step 3: get matched bboxes
    ir_box = inf_gt[(inf_gt['frame'] == frame_id) & (inf_gt['id'] == best_pair[0])][['x', 'y', 'w', 'h']].iloc[0].values
    rgb_box = rgb_gt[(rgb_gt['frame'] == frame_id) & (rgb_gt['id'] == best_pair[1])][['x', 'y', 'w', 'h']].iloc[0].values

    M = compute_affine_align_matrix(ir_box, rgb_box)
    # Step 5: warp IR image
    h_rgb, w_rgb = img_rgb.shape[:2]
    aligned_ir = cv2.warpAffine(img_ir, M, (w_rgb, h_rgb),borderMode=cv2.BORDER_CONSTANT,borderValue=0)

    # Step 6: overlay IR onto RGB
    blended = cv2.addWeighted(img_rgb, 1, aligned_ir, 0, 0)

    # Step 7: draw matched boxes
    cv2.rectangle(blended, (int(rgb_box[0]), int(rgb_box[1])),
                  (int(rgb_box[0]+rgb_box[2]), int(rgb_box[1]+rgb_box[3])), (0, 255, 0), 2)
    cv2.rectangle(blended, (int(M[0, 0]*ir_box[0]+M[0, 2]), int(M[1, 1]*ir_box[1]+M[1, 2])),
                  (int(M[0, 0]*(ir_box[0]+ir_box[2])+M[0, 2]), int(M[1, 1]*(ir_box[1]+ir_box[3])+M[1, 2])), (0, 0, 255), 2)

    # Step 8: save output
    # output_path = "aligned_ir_to_rgb.jpg"
    # cv2.imwrite(output_path, blended)

    #show result
    cv2.imshow("aligned_ir", aligned_ir)
    cv2.imshow("blended", blended)
    cv2.waitKey(0)
cv2.destroyAllWindows()