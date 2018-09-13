import numpy as np
import cv2
import argparse
from collections import deque
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from scipy import misc

from sklearn.externals import joblib
from skimage.feature import hog


cap=cv2.VideoCapture(0)

pts = deque()

Lower_green = np.array([110,50,50])
Upper_green = np.array([130,255,255])
while True:
    ret, img=cap.read()
    imgDraw = np.zeros(img.shape,np.uint8)
    img=cv2.flip(img,1)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    kernel=np.ones((5,5),np.uint8)
    mask=cv2.inRange(hsv,Lower_green,Upper_green)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    #mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    res=cv2.bitwise_and(img,img,mask=mask)
    cnts,heir=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    center = None
 
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
 
        if radius > 5:
            cv2.circle(img, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(img, center, 5, (0, 0, 255), -1)
        
    pts.appendleft(center)
    for i in range(1,len(pts)):
        if pts[i-1]is None or pts[i] is None:
            continue
        thick = int(np.sqrt(len(pts) / float(i + 1)) * 2.5)
        cv2.line(img, pts[i-1],pts[i],(0,0,225),5)
        cv2.line(imgDraw, pts[i-1],pts[i],(225,225,225),15)
        
    
    cv2.imshow("Frame", img)
    #cv2.imshow("mask",mask)
    #cv2.imshow("res",res)
    cv2.imwrite('num.jpg',imgDraw)

    
    
    k=cv2.waitKey(30) & 0xFF
    if k==32:
        break
# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
################################################################
# digits = load_digits()
# features = digits.data
# labels = digits.target

# clf=SVC(gamma=0.001)
# clf.fit(features,labels)

# img=misc.imread("num.jpg")
# img=misc.imresize(img,(8,8))
# img=img.astype(digits.images.dtype)
# img=misc.bytescale(img,high=16,low=0)
# x_test=[]
# for eachRow in img:
#     for eachPixel in eachRow:
#         x_test.append(sum(eachPixel)/3.0)
# print(clf.predict([x_test]))



# Load the classifier
clf = joblib.load("digits_cls.pkl")

# Read the input image 
im = cv2.imread("num.jpg")
im = cv2.bitwise_not(im)


# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
 
# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
ret,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    # Calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    print(nbr[0])

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()