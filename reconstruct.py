import cv2
import numpy as np

# Camera Matrix for barrel distortion
fx = 800
fy = 800
cx = 400 
cy = 250
k1 = -0.2
k2 = 0.01
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