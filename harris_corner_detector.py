import numpy as np 
from scipy import ndimage
import cv2


def get_grads(img):
	grad_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
	grad_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
	return grad_x, grad_y

def corner_detector(image, k=0.06, patch_size=3, threshold=0):
	"""
	input parameter: grayscale image
	output: image with corner features and corners [x,y, R]
	"""
	#show input grayscale image
	#cv2.imshow('Image', image)

	#calculate image intensity gradients
	I_x, I_y = get_grads(image)
	#calculate the matrix elements:
	I_xx = I_x**2
	I_xy = I_x*I_y
	I_yy = I_y**2

	corners = []
	new_img = image.copy()
	new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)

	summing_kernel = np.ones((patch_size, patch_size))

	A_mat = ndimage.convolve(I_xx, summing_kernel, mode='constant')
	B_mat = ndimage.convolve(I_xy, summing_kernel, mode='constant')
	C_mat = ndimage.convolve(I_yy, summing_kernel, mode='constant')

	determinants = (A_mat * C_mat) - B_mat**2
	traces = A_mat + C_mat
	R_matrix = determinants - k*(traces)**2
	
	threshold = np.max(R_matrix)/1e3
	print('threshold: ', threshold)
	for x in range(R_matrix.shape[0]):
		for y in range(R_matrix.shape[1]):
			if R_matrix[x, y] > threshold:
				corners.append((R_matrix[x, y], x, y))
				new_img.itemset((x, y, 0), 0)
				new_img.itemset((x, y, 1), 0)
				new_img.itemset((x, y, 2), 255)
	return new_img, corners

