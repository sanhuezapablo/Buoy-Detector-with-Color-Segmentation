import cv2
import numpy as np
import os
import math
import copy
#cap=cv2.VideoCapture('detectbuoy.avi')
pressed=False
refPt=[]

def click(event, x, y, flags, param):
	global pressed,refPt	
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt.append([x, y])
		cv2.circle(image,(x,y),12,(0,255,0))
		pressed=True
	elif event == cv2.EVENT_LBUTTONUP:
		pressed=False
		cv2.imshow("image", image)

for i in range(0,42):

	image=cv2.imread('Frames/frame%d.jpg' %i)
	clone=image.copy()
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", click)    

	while True:
		# display the image and wait for a keypress
		cv2.imshow("image", image)
		key = cv2.waitKey(1) & 0xFF
	 
		# if the 'r' key is pressed, reset the cropping region
		if key == ord("r"):
			image = clone.copy()

		elif key == ord("c"):
			break

	r=8
	x1=refPt[0][0]
	y1=refPt[0][1]
	x2=refPt[1][0]
	y2=refPt[1][1]
	x3=refPt[2][0]
	y3=refPt[2][1]

	roi1 = image[ y1-r:y1+r,x1-r:x1+r]
	roi2 = image[ y2-r:y2+r,x2-r:x2+r]
	roi3 = image[ y3-r:y3+r,x3-r:x3+r]

	cv2.imshow("ROI", roi1)
	cv2.imshow("ROI", roi2)
	cv2.imshow("ROI", roi3)

	cv2.imwrite("Yellow/yellow_%d.jpg" % i,roi1)
	cv2.imwrite("Red/red_%d.jpg" % i,roi2)
	cv2.imwrite("Green/green_%d.jpg" % i,roi3)

	refPt=[]

cv2.waitKey(0)
cv2.destroyAllWindows()