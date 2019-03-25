import numpy as np 
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt

def get_grads(img):
	kernel_x = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape(3, 3)
	kernel_y = kernel_x.transpose()
	grad_x = ndimage.convolve(img, kernel_x, mode='constant')
	grad_y = ndimage.convolve(img, kernel_y, mode='constant')
	return grad_x, grad_y

def corner_detector(image, k=0.06, patch_size=3, threshold=0):
	"""
	input parameter: grayscale image
	output: image with corner features and corners [x,y, R]
	"""
	#show input grayscale image
	#cv2.imshow('Image', image)

	#calculate image intensity gradients
	# I_x, I_y = np.gradient(image)
	#I_x, I_y = get_grads(image)

	I_x = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)
	I_y = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)

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
