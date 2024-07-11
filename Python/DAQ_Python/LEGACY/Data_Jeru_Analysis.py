location = "D:/Users/Manips/Documents/DATA/FRICS/Data_Jeru_ForYohann/data_2/data_2_fit.mat"
import numpy as np

try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *


dict=import_struc_from_matlab(location)


# ###
# import matplotlib.pyplot as plt
# plt.plot(dict["Uxy"][:,0])
# plt.show()
###

#ch_1,ch_2,ch_3=V_to_strain(dict["U1"],amp=,G=,i_0=,R=),V_to_strain(dict["U2"]),V_to_strain(dict["U3"],G=1.86)



U1=dict["U1"].transpose().reshape(-1)
U2=dict["U2"].transpose().reshape(-1)
U3=dict["U3"].transpose().reshape(-1)

Uxx,Uyy,Uxy=rosette_to_tensor(U1,U2,U3)

Uxx=Uxx.reshape(18,13000).transpose()
Uxy=Uxy.reshape(18,13000).transpose()
Uyy=Uyy.reshape(18,13000).transpose()

print(np.max(np.abs(Uxx-dict["Uxx"])))
print(np.max(np.abs(Uxy-dict["Uxy"])))
print(np.max(np.abs(Uyy-dict["Uyy"])))

###
location = "D:/Users/Manips/Documents/DATA/FRICS/Data_Jeru_ForYohann/"
for i in range(18):
    results=np.array([dict['t'],dict['Uxx'],dict['Uyy'],dict['Uxy']])/1000
    np.save(location+'gauge_{}_epsilon_time_xx_yy_xy.npy'.format(i),results)




