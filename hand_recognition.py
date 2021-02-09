import cv2
import numpy as np
import math

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.VideoCapture(0)
img.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
img.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
arr = []

while True:
	ret, frame = img.read()
	frame1 = cv2.flip(frame, 180)
	gray_img = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
	
	faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
	for (x,y,w,h) in faces:cv2.rectangle(frame1, (x-50,y-50), (x+w+50,y+h+50), (0,0,0), -1)
	
	hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
	lower_range = np.array([0,40,60])
	upper_range = np.array([30,255,255])
	
	mask = cv2.inRange(hsv, lower_range, upper_range)
	mask = cv2.erode(mask, None, iterations= 2)
	mask = cv2.dilate(mask, None, iterations=2)
	res = cv2.bitwise_and(frame1, frame1, mask = mask)
	
	contour, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	contours = max(contour, key = lambda x: cv2.contourArea(x))
	#approximating contour
	eps = 0.0005*cv2.arcLength(contours, True)
	approx = cv2.approxPolyDP(contours, eps, True)

	hull = cv2.convexHull(contours)

	hull_area = cv2.contourArea(hull)
	cnt_area = cv2.contourArea(contours)

	area_ratio = ((hull_area-cnt_area)/cnt_area)*100
	#finding defects
	hull = cv2.convexHull(approx,returnPoints=False)
	defects = cv2.convexityDefects(approx, hull)

	no_defects = 0

	#finding number of defects
	for i in range(defects.shape[0]):
		s,e,f,d = defects[i,0]
		start = tuple(approx[s][0])
		end = tuple(approx[e][0])
		farthest = tuple(approx[f][0])
		p = (100,180)

		a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
		b = math.sqrt((farthest[0] - start[0])**2 + (farthest[1] - start[1])**2)
		c = math.sqrt((end[0] - farthest[0])**2 + (end[1] - farthest[1])**2)
		s = (a+b+c)/2
		area = math.sqrt(s*(s-a)*(s-b)*(s-c))

		d = (2*area)/a
		#cosine rule
		angle = math.acos((b**2 + c**2 - a**2)/(2*b*c))*57

		if angle<=90 and d>30:
			no_defects+=1
			cv2.circle(res, farthest,5,[255,255,0],-1)
	no_defects+=1
	

	font = cv2.FONT_HERSHEY_SIMPLEX
	if no_defects==1:
		Top = tuple(contours[contours[:,:,1].argmin()][0])

		if cnt_area>=2000:
			if area_ratio<12:

				cv2.putText(frame1, '0', (0,50), font, 2, (255,255,255), 3, cv2.LINE_AA)
				arr= []
				# for i in range(len(arr)):
				# 	if(arr[i]==Top):

				# 		ax.append(i)
				# #for i in ax:
				# 	#print(i) 
				# 	#arr.pop(i)

			else:
				
				cv2.putText(frame1, '1', (0,50), font, 2, (255,255,255), 3, cv2.LINE_AA)			

	

	arr.append(tuple(contours[contours[:,:,1].argmin()][0]))
	cv2.rectangle(frame1, (500,0), (1280,500), (0,0,255), 3)

	for i in arr:
		cv2.circle(frame1, i, 8, (255,0,0), -1)
	

	#cv2.drawContours(res, [hull], -1, (255,0,0), 2)
	
	cv2.imshow("res", res)
	cv2.imshow("mask", mask)
	cv2.imshow("final", frame1)
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break
	if ret == False:
		break

cv2.destroyAllWindows()		
img.release()