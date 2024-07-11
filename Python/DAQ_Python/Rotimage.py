"""
Just a simple tool to rotate images
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def rotate_image(location,alpha,suf="_rot"):
    colorImage  = Image.open(location)
    rotated     = colorImage.rotate(alpha*180/np.pi)
    i = location[::-1].find(".")
    location_out = location[:-i-1]+suf+location[-i-1:]
    rotated.save(location_out)
    return(None)


def rotate_directory(directory,alpha,suf="_rot"):
    if directory[-1] not in ["\\","/"] :
        directory+='/'
    lsdir=sorted(os.listdir(directory))
    if isinstance(alpha,(list,tuple,np.ndarray)):
        if len(alpha)==len(lsdir):
            for i in range(len(lsdir)):
                try:
                    rotate_image(directory+lsdir[i],alpha[i],suf)
                except:
                    print("Error during the rotation of {}, probably not an image supported by PIL.".format(lsdir[i]))
        else:
            raise("len(alpha) != number of elements in directory")
    else:
        for location in lsdir:
            try:
                rotate_image(directory+location,alpha,suf)
            except:
                print("Error during the rotation of {}, probably not an image supported by PIL.".format(location))
    return(None)

## Application
"""
directory = "D:/Users/Manips/Documents/DATA/FRICS/2021-02-02/paliers/"
alpha=1

rotate_directory(directory,[0.1,0.2,0.3,0.4])
"""