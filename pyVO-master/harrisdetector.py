import cv2
import numpy as np
from operator import itemgetter
from typing import Tuple, List


def get_grads(img):
	grad_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
	grad_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
	# grad_x = cv2.Scharr(img,cv2.CV_64F,1,0)
	# grad_y = cv2.Scharr(img,cv2.CV_64F,0,1)
	return grad_x, grad_y


def harris_corners(image: np.ndarray, patch_size=3, threshold_div_factor=1e4, blur_sigma=2.0) -> List[Tuple[float, np.ndarray]]:
	"""
	Return the harris corners detected in the image.

	:param img:				 The grayscale image.	shape (480, 640)
	:param threshold:		 The harris respnse function threshold.
	:param blur_sigma: 		 Sigma value for the image bluring.

	:return:
					A sorted list of tuples containing response value and image position.
			        The list is sorted from largest to smallest response value.
	"""

	"""
	input parameter:		 grayscale image
	output: 				 image with corner features and corners ( list of tuples(float, np.array) )
	"""

	#calculate image intensity gradients
	I_x, I_y = get_grads(image)
	#calculate the matrix elements:
	I_xx = I_x**2
	I_xy = I_x*I_y
	I_yy = I_y**2

	corners = []
	new_img = image.copy()
	#new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB) trenger ikke denne

	summing_kernel = np.ones((patch_size, patch_size))

	A_mat = cv2.filter2D(I_xx, -1, summing_kernel)
	B_mat = cv2.filter2D(I_xy, -1, summing_kernel)
	C_mat = cv2.filter2D(I_yy, -1, summing_kernel)

	determinants = (A_mat * C_mat) - B_mat**2
	traces = A_mat + C_mat
	k = 0.2
	R_matrix = determinants - k*(traces)**2

	threshold = np.max(R_matrix)/threshold_div_factor

	corner_indices = np.argwhere(R_matrix>threshold)

	for x, y in corner_indices:
		# corners = list of tuples(float, np.ndarray)
		corners.append(( R_matrix[x, y], np.array([x, y]) ))
		#new_img.itemset((x, y, 0), 0)
		#new_img.itemset((x, y, 1), 0)				trenger ikke disse
		#new_img.itemset((x, y, 2), 255)
	corners = sorted(corners, key=lambda x : x[0], reverse=True)


	# need only return list of corners
	return corners
