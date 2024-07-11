"""
comparision between before and after an experiment
"""

location="D:/Users/Manips/Documents/DATA/FRICS/2022/2022-05-11/before_after/"

import numpy as np

before_loading=np.loadtxt(location+"{}_loading{}.txt".format("before",""))
before_loading_jeru=np.loadtxt(location+"{}_loading{}.txt".format("before","_jeru"))
after_loading=np.loadtxt(location+"{}_loading{}.txt".format("after",""))
after_loading_jeru=np.loadtxt(location+"{}_loading{}.txt".format("after","_jeru"))

print("Our mean and std")
print("{:.2f}mV".format(np.abs(before_loading-after_loading).mean()*1000))
print("{:.2f}mV".format(np.abs(before_loading-after_loading).std()*1000))

print("\nJerusalems 'mean and std")
print("{:.2f}mV".format(np.abs(before_loading_jeru-after_loading_jeru).mean()*1000))
print("{:.2f}mV".format(np.abs(before_loading_jeru-after_loading_jeru).std()*1000))
