import cv2
import numpy as np

# Đọc ảnh
img1 = cv2.imread(r'D:\inProgress\DeepLearning\Project\PIDNet\data\ADE20K\images\validation\ADE_val_00000003.jpg')
img2 = cv2.imread(r'D:\inProgress\DeepLearning\Project\PIDNet\data\ADE20K\annotations\validation\ADE_val_00000003.png')

# Resize về cùng chiều cao (nếu cần)
h = min(img1.shape[0], img2.shape[0])
img1 = cv2.resize(img1, (int(img1.shape[1] * h / img1.shape[0]), h))
img2 = cv2.resize(img2, (int(img2.shape[1] * h / img2.shape[0]), h))

# Ghép ngang
result = np.hstack((img1, img2))

# Lưu hoặc hiển thị
cv2.imwrite('merged_horizontal.jpg', result)