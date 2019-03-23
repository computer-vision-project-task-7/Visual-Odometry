import numpy as np 
from scipy import signal
import cv2

def corner_detector(image, k=0.06, patch_size=3, threshold=0):
	"""
	input parameter: grayscale image
	output: image with corner features and corners [x,y, R]
	"""
	#calculate image intensity gradients
	I_x, I_y = np.gradient(image)
	#calculate the matrix elements:
	I_xx = I_x**2
	I_xy = I_x*I_y
	I_yy = I_y**2

	height, width = image.shape
	corners = []
	offset = int(patch_size/2)
	new_img = image.copy()
	new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)

	summing_kernel = np.ones((patch_size, patch_size))

	A_mat = signal.convolve2d(I_xx, summing_kernel, mode='same')
	B_mat = signal.convolve2d(I_xy, summing_kernel, mode='same')
	C_mat = signal.convolve2d(I_yy, summing_kernel, mode='same')

	determinants = (A_mat * C_mat) - B_mat**2
	traces = A_mat + C_mat
	R_matrix = determinants - k*(traces**2) 

	for y in range(R_matrix.shape[0]):
		for x in range(R_matrix.shape[1]):
			if R_matrix[y, x] > threshold:
				corners.append([x, y, R_matrix[y, x]])
				new_img.itemset((y, x, 0), 0)
				new_img.itemset((y, x, 1), 0)
				new_img.itemset((y, x, 2), 255)

	return new_img, corners

	###############Old code below, much slower, do not use####################

	
	# for y in range(offset, height - offset):
	# 	for x in range(offset, width - offset):
	# 		A = np.sum(I_xx[(y-offset):(y+offset+1), (x-offset):(x+offset+1)])
	# 		C = np.sum(I_yy[(y-offset):(y+offset+1), (x-offset):(x+offset+1)])
	# 		B = np.sum(I_xy[(y-offset):(y+offset+1), (x-offset):(x+offset+1)])

	# 		determinant = (A * C) - B**2
	# 		trace = A + C
	# 		R = determinant - k*(trace**2)
	# 		if R > 0:
	# 			corners.append([x, y, R])
	# 			new_img.itemset((y, x, 0), 0)
	# 			new_img.itemset((y, x, 1), 0)
	# 			new_img.itemset((y, x, 2), 255)
	# return new_img, corners