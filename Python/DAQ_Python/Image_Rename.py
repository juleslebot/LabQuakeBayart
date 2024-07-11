import os
import numpy as np


def Image_Rename(directory,prefix="img_"):
    lsdir=os.listdir(directory)
    i=0
    if directory[-1] not in ["\\","/"] :
        directory+='/'
    for location in sorted(lsdir):
        j = location[::-1].find(".")
        ext = location[-j-1:]
        os.rename(directory+location,directory+prefix+str(i)+ext)
        i+=1
    return(None)