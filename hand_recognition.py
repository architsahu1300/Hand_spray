import cv2
import numpy as np

img = cv2.VideoCapture(0)

while True:
	ret, frame = img.read()
	frame1 = cv2.flip(frame, 180)
	hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
	lower_range = np.array([0,40,55])
	upper_range = np.array([20,255,255])
	
	mask = cv2.inRange(hsv, lower_range, upper_range)
	res = cv2.bitwise_and(frame1, frame1, mask = mask)
	
	contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	contours = max(contours, key = lambda x: cv2.contourArea(x))
	
	hull = cv2.convexHull(contours)
	

	cv2.drawContours(res, [hull], -1, (255,0,0), 2)
	
	cv2.imshow("mask", res)
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break
	if ret == False:
		break

cv2.destroyAllWindows()		
img.release()