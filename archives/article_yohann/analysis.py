import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import fnmatch
import scipy.io as scio
import glob



locals().update(np.load('C:/Users/Manips/Desktop/tmp/locals.npy',allow_pickle=True).all())

##



def smooth(x, n=50):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    rolling_avg = (cumsum[n:] - cumsum[:-n]) / float(n)
    return(rolling_avg)


def plot_fig(sig_xy,sig_yy,ante,res,inert):
    fig, axes = plt.subplots(10, 2, sharex=True,sharey="col",gridspec_kw={'hspace': 0})
    # Plot each row of sig_xy and sig_yy on corresponding subplots
    for i in range(10):
        # Plot sig_xy on the left column
        axes[i, 0].plot(smooth(sig_xy[i],50))

        # Plot sig_yy on the right column
        axes[i, 1].plot(smooth(sig_yy[i],50))

        axes[i,0].axhline(res[1][i],c="k")
        axes[i,0].axhline(ante[1][i],c="r")
        axes[i,0].axhline(inert[1][i],c="g")
        axes[i,1].axhline(res[0][i],c="k")
        axes[i,1].axhline(ante[0][i],c="r")
        axes[i,1].axhline(inert[0][i],c="g")
        axes[i,0].grid(which='both')
        axes[i,1].grid(which='both')

    if i < 9:
        axes[i, 0].tick_params(labelbottom=False)
        axes[i, 1].tick_params(labelbottom=False)


    # Adjust layout to prevent overlap
    plt.tight_layout()
    print(i,j)
    # Display the plot
    plt.show()


##

"""
for i in range(len(sig_full)):
    sig_full_manip=sig_full[i]
    for j in range(len(sig_full_manip)):
        sig = sig_full_manip[j]
        sig_xy=sig[2::3]
        sig_yy=sig[1::3]
        res,ante,inert = residual[i][j],antecedent[i][j],inertial[i][j]
        plot_fig(sig_xy,sig_yy,ante,res, inert)
        plt.show()
        input()
"""


x_plot = [1, 2, 3, 4, 5, 8, 11, 12, 13, 14]






#for i in range(len(sig_full)):

for i in [2]:

    fig,axes = plt.subplots(1,3,sharex = True, sharey=True)


    sig_full_manip=sig_full[i]
    for j in range(len(sig_full_manip)):
        sig = sig_full_manip[j]
        sig_xy=sig[2::3]
        sig_yy=sig[1::3]
        res,ante,inert = residual[i][j],antecedent[i][j],inertial[i][j]
    #   plot_fig(sig_xy,sig_yy,ante,res, inert)
    #   plt.show()

        res_prof = res[1]/res[0]
        ante_prof = ante[1]/ante[0]
        inert_prof = inert[1]/inert[0]

        axes[0].plot(x_plot,ante_prof, c="k", alpha = .5, marker = "+")
        axes[1].plot(x_plot,res_prof, c="k", alpha = .5, marker = "+")
        axes[2].plot(x_plot,inert_prof, c="k", alpha = .5, marker = "+")


    axes[0].set_ylabel(r"$\sigma_{xy}/\sigma_{yy}$")
    axes[1].set_xlabel(r"$x$ (cm)")


    for ax in axes:
        ax.grid(True)

    axes[0].set_title("Chargé")
    axes[1].set_title("Residuel")
    axes[2].set_title("Inertiel")
    fig.suptitle(r"$C_\sigma = {:.2f}$".format(np.mean(loading_contrast[i])))

    plt.show()

##

no_hole = [0,1,2,3,4,6,7,8,9]

fig,axes = plt.subplots(1,3,sharex = True, sharey=True)

markers = ['s','.','^','*','v','p']
colors = ['C0','C1','C2','C3','C4','C5']

for i in range(len(sig_full)):
    m = markers[(i+1)%6]
    c = colors[i//6-1]
    sig_full_manip=sig_full[i]
    for j in range(len(sig_full_manip)):
        res,ante,inert = residual[i][j],antecedent[i][j],inertial[i][j]

        axes[0].plot(ante[0][no_hole ],ante[1][no_hole ],alpha = .2,marker=m,c=c)
        axes[1].plot(res[0][no_hole ],res[1][no_hole ],alpha = .2,marker=m,c=c)
        axes[2].plot(inert[0][no_hole ],inert[1][no_hole ],alpha = .2,marker=m,c=c)

        axes[0].plot(ante[0][5 ],ante[1][5 ],'.',c="b",alpha = .2)
        axes[1].plot(res[0][5 ],res[1][5 ],'.',c="b",alpha = .2)
        axes[2].plot(inert[0][5 ],inert[1][5 ],'.',c="b",alpha = .2)



fig.show()














