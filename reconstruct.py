import cv2
import numpy as np

fx = 822.79041
fy = 822.79041
cx = 313.47335
cy = 231.31299
k1 = -0.28341
k2 = 0.07374
p1 = 0.00000
p2 = 0.00000
k3 = 0.00000

camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])
dist_coeffs = np.array([k1, k2, p1, p2, k3])

image = cv2.imread('img/ahalf.png')

new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, image.shape[:2], 1)

undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
cv2.imwrite('img/undistorted.png', undistorted_image)

cv2.imshow('Undistorted Image', undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()