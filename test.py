import cv2
from harris_corner_detector import corner_detector
import time
import numpy as np


img = cv2.imread('images/skyline_toronto.jpg', 0)

#resize for speed, change the values for fx and fy to scale. e.i fx=fy=0.5.
scale_factor = 1
img = cv2.resize(img, (0,0), fx=scale_factor, fy=scale_factor) 

# cols, rows = img.shape
# M = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
# img = cv2.warpAffine(img,M,(cols,rows))
# cv2.imwrite('rotated_chessboard.png', img)

print('Scale factor: {}, Shape of scaled image: {}'.format(scale_factor, img.shape))

t = time.time()
img_corner, corners = corner_detector(img, k=0.06, patch_size=3, threshold=0)
print("Corner detection took: ", time.time()-t)
print("FPS: ", 1/(time.time()-t))

#rescale to original size.
img_corner = cv2.resize(img_corner, (0,0), fx=1/scale_factor, fy=1/scale_factor)

cv2.imshow('img', img_corner)
cv2.waitKey(0)

#Save image if needed.
#cv2.imwrite('filename.jpg', img_corner)