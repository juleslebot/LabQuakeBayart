from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt



listAllCircles = loadmat('D:/Users/Manips/Documents/DATA/FRICS/2021-03-16/Subpixel2/listAllCircles.mat')["listAllCircles"][0]

listAllCircles=listAllCircles[[i for i in range(303) if i!=147]]

paliers = np.loadtxt('D:/Users/Manips/Documents/DATA/FRICS/2021-03-16/Subpixel2/paliers.txt')
pxtomm = 13.57



for i in range(302):
    plt.plot(paliers[:-1]-paliers[0],(listAllCircles[i]["xSub"].T-listAllCircles[i]["xSub"].T[0])*1000/pxtomm)

plt.show()
