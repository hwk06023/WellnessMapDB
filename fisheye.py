import cv2
import numpy as np

# Camera Matrix for barrel distortion
fx = 800
fy = 800
cx = 400
cy = 300

k1 = -0.2
k2 = 0.01
p1 = 0.00000
p2 = 0.00000
k3 = 0.00000

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])
D = np.array([k1, k2, p1, p2, k3])

image = cv2.imread('img/ahalf.png')

DIM = image.shape[:2][::-1]
balance = 0.0

dim1 = DIM
dim2 = tuple([int(x / 1.0) for x in DIM])
dim3 = tuple([int(x / 1.0) for x in DIM])

scaled_K = K * dim1[0] / DIM[0]

new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)

map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)

undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

cv2.imwrite('img/fisheye.png', undistorted_image)
cv2.imshow('Undistorted Image', undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()