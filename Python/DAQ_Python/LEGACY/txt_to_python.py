import numpy as np
import matplotlib.pyplot as plt


location='D:/Users/Manips/Documents/DATA/FRICS/2022/2022-01-26/'
time=np.loadtxt(location+'time.txt')/1000
u_1,u_2,u_3=np.loadtxt(location+'U1.txt'),np.loadtxt(location+'U2.txt'),np.loadtxt(location+'U3.txt')
u_1,u_2,u_3=u_1/1000,u_2/1000,u_3/1000



## transformations to restore tensor


def rosette_to_tensor(ch_1,ch_2,ch_3):
    """
    converts a 45 degres rosette signal into a full tensor.
    input : the three channels of the rosette
    output : $\epsilon_{xx},\epsilon_{yy},\epsilon_{xy}
    https://www.efunda.com/formulae/solid_mechanics/mat_mechanics/strain_gage_rosette.cfm
    """
    eps_xx=ch_1+ch_3-ch_2
    eps_yy=ch_2
    eps_xy=(ch_1-ch_3)/2
    return(eps_xx,eps_yy,eps_xy)


chans=[]
for i in range(u_1.shape[-1]):
    xx,yy,xy=rosette_to_tensor(u_1[:,i],u_2[:,i],u_3[:,i])
    chans.append([xx,yy,xy])
    results=np.array([time,xx,yy,xy])
    np.save(location+'npyfiles/chan {}'.format(i)+'_epsilon_time_xx_yy_xy.npy',results)
