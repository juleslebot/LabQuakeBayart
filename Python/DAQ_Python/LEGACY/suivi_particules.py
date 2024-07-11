import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import imutils

img_rgb = cv.imread('D:/Users/Manips/Desktop/cine3_full.png')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('D:/Users/Manips/Desktop/template_gros.png',0)

resized = imutils.resize(template, width = int(template.shape[1] * 0.9))

w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
res_2=cv.matchTemplate(img_gray,resized,cv.TM_CCOEFF_NORMED)

threshold = 0.5


loc_1 = np.where( res >= threshold)
loc_2 = np.where( res_2 >= threshold)

loc=(np.append(loc_1[0],loc_2[0]),np.append(loc_1[1],loc_2[1]))

pts=list(zip(*loc[::-1]))


def distance(pt1,pt2):
    return()


from scipy.spatial.distance import cdist
dist = cdist(pts, pts)

rm=[]

for i in range(len(dist)-1):
    for j in range(i+1,len(dist)):
        if dist[i,j]<15 and j not in rm :
            rm.append(j)

rm=sorted(rm)[::-1]

for i in rm:
    _=pts.pop(i)

for pt in pts:
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
cv.imwrite('D:/Users/Manips/Desktop/res.png',img_rgb)