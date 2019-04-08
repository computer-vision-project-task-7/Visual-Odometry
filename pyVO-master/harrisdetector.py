import cv2
import numpy as np
from operator import itemgetter
from typing import Tuple, List


def get_grads(img):
	# grad_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
	# grad_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
	grad_x = cv2.Scharr(img,cv2.CV_64F,1,0)
	grad_y = cv2.Scharr(img,cv2.CV_64F,0,1)
	return grad_x, grad_y


def harris_corners(image: np.ndarray, patch_size=3, threshold_div_factor=1e3, k=0.06, blur_sigma=3.0) -> List[Tuple[float, np.ndarray]]:
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
	# summing_kernel = np.ones((patch_size, patch_size))
	# A_mat = cv2.filter2D(I_xx, -1, summing_kernel)
	# B_mat = cv2.filter2D(I_xy, -1, summing_kernel)
	# C_mat = cv2.filter2D(I_yy, -1, summing_kernel)
	A_mat = cv2.GaussianBlur(I_xx, (patch_size, patch_size), blur_sigma)
	B_mat = cv2.GaussianBlur(I_xy, (patch_size, patch_size), blur_sigma)
	C_mat = cv2.GaussianBlur(I_yy, (patch_size, patch_size), blur_sigma)

	determinants = (A_mat * C_mat) - B_mat**2
	traces = A_mat + C_mat
	R_matrix = determinants - k*(traces)**2

	threshold = np.max(R_matrix)/threshold_div_factor
	corner_indices = np.argwhere(R_matrix>threshold)
	# num tiles in each direction
	num_tiles = 10
	# heigth, width of each tile
	height = image.shape[0]//num_tiles #48
	width = image.shape[1]//num_tiles#64
	best_indices = []
	# splitting into 100 tiles
	for i in range(10):		#tile i høyden
		for j in range(10):	#tile i bredden
			# aktuell bit av R
			R_bit =  R_matrix[i*width:(i+1)*width , j*height:(j+1)*height]
			above = []
			# indices i biten av R som er over threshold
			above_threshold =  np.argwhere(R_bit > threshold )
			for v, u in above_threshold:
				# append R-verdi og pixel-kooridnatene til bildet til alle over threhsold
				above.append( (R_bit[v, u], (j*height + u,  i*width + v) ) )


			if len(above) != 0:
				#if it found any R-values above threshold
				best = sorted(above, key=lambda x : x[0], reverse=True)
				# append pixelkoordinaten til punkt med beste R-verdi i tilen
				best_indices.append(best[0][1])
				corners.append( (R_bit[v, u], best[0][1]) )

	corners = sorted(corners, key=lambda x : x[0], reverse=True)
	return corners[:80]
