import numpy as np 
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

	for y in range(offset, height - offset):
		for x in range(offset, width - offset):
			A = np.sum(I_xx[(y-offset):(y+offset+1), (x-offset):(x+offset+1)])
			C = np.sum(I_yy[(y-offset):(y+offset+1), (x-offset):(x+offset+1)])
			B = np.sum(I_xy[(y-offset):(y+offset+1), (x-offset):(x+offset+1)])

			determinant = (A * C) - B**2
			trace = A + C
			R = determinant - k*(trace**2)
			if R > 0:
				corners.append([x, y, R])
				new_img.itemset((y, x, 0), 0)
				new_img.itemset((y, x, 1), 0)
				new_img.itemset((y, x, 2), 255)
	return new_img, corners