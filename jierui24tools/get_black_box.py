import cv2
import numpy as np

def detect_black_boxes(image_path, area_thresh=500):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 提取黑色区域（近黑色像素值阈值 < 30）
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)

    # 查找黑色区域轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    vis_img = img.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > area_thresh:
            boxes.append((x, y, w, h))
            # 可视化绘制红色矩形框
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return boxes, vis_img


# === 用法 ===
image_path = '000327.jpg'  # 你的图片路径
black_boxes, vis_img = detect_black_boxes(image_path)

# 显示图像（按任意键关闭窗口）
cv2.imshow("Detected Black Boxes", vis_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 可选：保存可视化结果
#cv2.imwrite("black_boxes_vis.jpg", vis_img)
