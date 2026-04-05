import cv2
import numpy as np

# Đọc ảnh
img1 = cv2.imread(r'D:\inProgress\DeepLearning\Project\PIDNet\data\ADE20K\testing\ADE_test_00000028.jpg')
img2 = cv2.imread(r'D:\inProgress\DeepLearning\Project\PIDNet\data\ADE20K\testing\outputs\ADE_test_00000028.jpg')

# Resize về cùng chiều cao (nếu cần)
# h = min(img1.shape[0], img2.shape[0])
# img1 = cv2.resize(img1, (int(img1.shape[1] * h / img1.shape[0]), h))
# img2 = cv2.resize(img2, (int(img2.shape[1] * h / img2.shape[0]), h))

# Ghép ngang
# result = np.hstack((img1, img2))

# # Lưu hoặc hiển thị
# cv2.imwrite('Segmentation_Results.jpg', result)

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

# Chiều cao = max, chiều rộng = tổng
canvas_h = max(h1, h2)
canvas_w = w1 + w2

# Tạo nền đen
canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

# Đặt ảnh 1 bên trái
canvas[0:h1, 0:w1] = img1

# Đặt ảnh 2 bên phải
canvas[0:h2, w1:w1+w2] = img2

cv2.imwrite('Segmentation_Results.jpg', canvas)