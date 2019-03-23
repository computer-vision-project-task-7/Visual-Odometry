import cv2
from harris_corner_detector import corner_detector

img = cv2.imread('images/chessboard.jpg', 0)
#resize for speed, change the values for fx and fy to scale. e.i fx=fy=0.5
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
print(img.shape)

img_corner, corners = corner_detector(img, k=0.06, patch_size=3)

cv2.imshow('img', img_corner)
cv2.waitKey(0)

#cv2.imwrite('out_img.jpg', img_corner)