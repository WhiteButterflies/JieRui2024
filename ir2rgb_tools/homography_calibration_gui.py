# import cv2
# import numpy as np
#
# # 路径：用户替换为自己的图像路径
# rgb_image_path = 'rgb_000014.jpg'
# ir_image_path = 'ir_000014.jpg'
# H_SAVE_PATH = '0061_ir_to_rgb_h.npy'
#
# # 存储点坐标
# rgb_points = []
# ir_points = []
#
# # 当前图像状态（点击哪个）
# current_window = 'RGB'  # 或 'IR'
#
#
# def click_event(event, x, y, flags, param):
#     global current_window
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if current_window == 'RGB':
#             rgb_points.append([x, y])
#             print(f"RGB 点: ({x}, {y})")
#         elif current_window == 'IR':
#             ir_points.append([x, y])
#             print(f"IR 点: ({x}, {y})")
#     elif event == cv2.EVENT_RBUTTONDOWN:
#         if len(rgb_points) >= 4 and len(rgb_points) == len(ir_points):
#             rgb_np = np.array(rgb_points, dtype=np.float32)
#             ir_np = np.array(ir_points, dtype=np.float32)
#             H, _ = cv2.findHomography(ir_np, rgb_np, method=cv2.RANSAC)
#             np.save(H_SAVE_PATH, H)
#             print(f"保存单应性矩阵至 {H_SAVE_PATH}")
#         else:
#             print("点数不足或不一致，请至少配对4个点，并保持数量一致")
#
#
# def draw_points(image, points):
#     for (x, y) in points:
#         cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
#     return image
#
#
# def main():
#     global current_window
#
#     rgb_img = cv2.imread(rgb_image_path)
#     ir_img = cv2.imread(ir_image_path)
#
#     if rgb_img is None or ir_img is None:
#         print("读取图像失败，请检查路径。"); return
#
#     print("请在两个窗口中依次点击对应点，确保顺序一致（如：船头→桅杆→岸边等）。")
#     print("先点 RGB，再切换 IR，依次进行。按 Tab 切换窗口，右键点击任意窗口保存匹配矩阵，ESC 退出。")
#
#     while True:
#         rgb_copy = draw_points(rgb_img.copy(), rgb_points)
#         ir_copy = draw_points(ir_img.copy(), ir_points)
#
#         cv2.imshow('RGB', rgb_copy)
#         cv2.imshow('IR', ir_copy)
#
#         cv2.setMouseCallback('RGB', click_event)
#         cv2.setMouseCallback('IR', click_event)
#
#         key = cv2.waitKey(50) & 0xFF
#         if key == 27:  # ESC 退出
#             break
#         elif key == 9:  # Tab 切换窗口
#             current_window = 'IR' if current_window == 'RGB' else 'RGB'
#             print(f"切换窗口为：{current_window}")
#
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     main()
import cv2
import numpy as np

# 路径：用户替换为自己的图像路径
rgb_image_path = 'rgb_000164.jpg'
ir_image_path = 'ir_000164.jpg'
H_SAVE_PATH = '000164_ir_to_rgb_h.npy'

# 窗口显示大小（限制图像显示尺寸）
window_width = 1000  # 可调为你屏幕的一半宽度
window_height = 740

# 存储点坐标
rgb_points = []
ir_points = []

# 当前图像状态（点击哪个）
current_window = 'RGB'  # 或 'IR'

def resize_with_aspect_ratio(image, width, height):
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    return resized, scale

def click_event(event, x, y, flags, param):
    global current_window
    img, scale = param
    x_real, y_real = int(x / scale), int(y / scale)
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_window == 'RGB':
            rgb_points.append([x_real, y_real])
            print(f"RGB 点: ({x_real}, {y_real})")
        elif current_window == 'IR':
            ir_points.append([x_real, y_real])
            print(f"IR 点: ({x_real}, {y_real})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(rgb_points) >= 4 and len(rgb_points) == len(ir_points):
            rgb_np = np.array(rgb_points, dtype=np.float32)
            ir_np = np.array(ir_points, dtype=np.float32)
            H, _ = cv2.findHomography(ir_np, rgb_np, method=cv2.RANSAC)
            np.save(H_SAVE_PATH, H)
            print(f"保存单应性矩阵至 {H_SAVE_PATH}")
        else:
            print("点数不足或不一致，请至少配对4个点，并保持数量一致")

def draw_points(image, points, scale):
    for (x, y) in points:
        cv2.circle(image, (int(x * scale), int(y * scale)), 5, (0, 255, 0), -1)
    return image

def main():
    global current_window

    rgb_img = cv2.imread(rgb_image_path)
    ir_img = cv2.imread(ir_image_path)

    if rgb_img is None or ir_img is None:
        print("读取图像失败，请检查路径。")
        return

    print("请在两个窗口中依次点击对应点，确保顺序一致（如：船头→桅杆→岸边等）。")
    print("先点 RGB，再切换 IR，依次进行。按 Tab 切换窗口，右键点击任意窗口保存匹配矩阵，ESC 退出。")

    while True:
        rgb_disp, rgb_scale = resize_with_aspect_ratio(rgb_img.copy(), window_width, window_height)
        ir_disp, ir_scale = resize_with_aspect_ratio(ir_img.copy(), window_width, window_height)

        rgb_disp = draw_points(rgb_disp, rgb_points, rgb_scale)
        ir_disp = draw_points(ir_disp, ir_points, ir_scale)

        cv2.imshow('RGB', rgb_disp)
        cv2.imshow('IR', ir_disp)

        cv2.setMouseCallback('RGB', click_event, param=(rgb_img, rgb_scale))
        cv2.setMouseCallback('IR', click_event, param=(ir_img, ir_scale))

        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # ESC 退出
            break
        elif key == 9:  # Tab 切换窗口
            current_window = 'IR' if current_window == 'RGB' else 'RGB'
            print(f"切换窗口为：{current_window}")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
