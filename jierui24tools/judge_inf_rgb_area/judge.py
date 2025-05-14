import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load GT
columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
inf_gt = pd.read_csv('gt.txt', names=columns)
rgb_gt = pd.read_csv('gt_mask.txt', names=columns)

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

frame_id = 20
inf_vecs = compute_motion_vector(inf_gt, frame_id)
rgb_vecs = compute_motion_vector(rgb_gt, frame_id)

matches = []
for iid, ivec in inf_vecs.items():
    best_id, best_sim = None, -1
    for rid, rvec in rgb_vecs.items():
        sim = cosine_similarity([ivec], [rvec])[0][0]
        if sim > best_sim:
            best_id, best_sim = rid, sim
    if best_sim > 0.9:
        matches.append((iid, best_id))

# 可视化
rgb_frame = rgb_gt[rgb_gt['frame'] == frame_id]
fig, ax = plt.subplots(figsize=(12, 8))
for _, rid in matches:
    box = rgb_frame[rgb_frame['id'] == rid]
    if not box.empty:
        x, y, w, h = box.iloc[0][['x', 'y', 'w', 'h']]
        ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none'))

ax.set_xlim(0, 1920)
ax.set_ylim(1080, 0)
plt.title(f'Matched RGB Boxes at Frame {frame_id}')
plt.show()
