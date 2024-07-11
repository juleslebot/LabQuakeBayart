### Analyse_stat.py
# The goal here is to make a statistical analysis of the results of expereiments
# We make nice histograms for many different variables

## Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import fnmatch


try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *



def list_files_with_pattern(directory, pattern):
    """
    List files in a directory which names contain a certain pattern
    """
    matching_files = []
    for file_name in os.listdir(directory):
        if fnmatch.fnmatch(file_name, f'*{pattern}*'):
            matching_files.append(file_name)
    return(matching_files)



def temp_nice_plot(y_min,x, ax = plt.gca() ) :
    """
    plots a nice representation of the bloc in an histogram
    Useful way later
    """
    ylim=ax.get_ylim()
    dilat = ylim[1]-ylim[0]
    ax.set_ylim((y_min-0.1*dilat,ylim[1]))
    import matplotlib.patches as patches
    arc_radius = 0.05*dilat
    arc_center_x = 0.075
    arc_center_y = y_min
    start_angle = 0
    end_angle = 180
    arc_patch = patches.Arc((arc_center_x, arc_center_y), width=0.03, height=2*arc_radius, angle=0,
                            theta1=start_angle, theta2=end_angle, color='r', linewidth=2)
    ax.add_patch(arc_patch)

    # lines of the block
    line1 = patches.ConnectionPatch((0, y_min), (0.06, y_min), "data", "data", edgecolor="r", linewidth=2, arrowstyle="-")
    ax.add_patch(line1)
    line2 = patches.ConnectionPatch((0.09, y_min), (0.15, y_min), "data", "data", edgecolor="r", linewidth=2, arrowstyle="-")
    ax.add_patch(line2)
    line3 = patches.ConnectionPatch((0, y_min), (0, y_min+0.25*dilat), "data", "data", edgecolor="r", linewidth=2, arrowstyle="-")
    ax.add_patch(line3)
    line4 = patches.ConnectionPatch((0.15, y_min), (0.15, y_min+0.25*dilat), "data", "data", edgecolor="r", linewidth=2, arrowstyle="-")
    ax.add_patch(line4)

    # gages, used and unused
    x_not = [0.005+0.01*i for i in range(15)]
    for xi in x_not:
        square_patch = patches.Rectangle((xi-0.002, y_min+0.07*dilat), 0.004, 0.04*dilat, color='grey',alpha=.3)
        ax.add_patch(square_patch)

    for xi in x:
        square_patch = patches.Rectangle((xi-0.002, y_min+0.07*dilat), 0.004, 0.04*dilat, color='r',alpha=1)
        ax.add_patch(square_patch)

    ax.set_xlim((-0.01,0.16))






## Data location
# location of the main folder containing all the "manip_..." subfolders


loc_folder="D:/Users/Manips/Documents/DATA/FRICS/2023/2023-07-11-manips-10-voies/"
loc_figures = loc_folder + "histograms/"
loc_manip = "manip_{}/"
loc_params="parameters.txt"
file_name = "event-0{:02d}.npy_times_hand_picked.npy"


if not os.path.exists(loc_figures):
    os.makedirs(loc_figures)


## Load all data

# load params
exec(load_params(loc_folder+loc_manip.format(1)+loc_params))
x=x-0.005
gages_channels = np.concatenate([np.arange(0,15),np.arange(16,31)])
nchannels = len(gages_channels)

# create main lists
times=[]
mus = []
eps_yys = []
eps_xys = []
loading_contrast = []
fns=[]
from_solid=[]

# List of experiments that are not clean for the stats.
to_remove = [7,19,20,21,22,23,24,25,32,33,41]
solids = [14,15,16,17,18,37,38]


manips=list_files_with_pattern(loc_folder,"manip")

for i in to_remove:
    manips.remove("manip_{}".format(i))

for manip in manips:
    print(manip)
    # create location name
    directory_path = loc_folder + manip + "/"
    pattern_to_match = "npy_times"
    matching_files = list_files_with_pattern(directory_path, pattern_to_match)
    j_max=len(matching_files)

    for file in matching_files :
        # times
        print(file)
        times_hand_picked = np.load(directory_path+file)
        times.append(times_hand_picked)

        # gages
        data = np.load(directory_path+file[:13])
        gages  = data[gages_channels]
        fn = data[32]
        gages_zero = np.load(directory_path+"event-001.npy")[gages_channels]
        gages=np.transpose(np.transpose(gages)-np.mean(gages_zero,axis=-1))
        for i in range(nchannels//3):
            ch_1=gages[3*i]
            ch_2=gages[3*i+1]
            ch_3=gages[3*i+2]
            ch_1,ch_2,ch_3=voltage_to_strains(ch_1,ch_2,ch_3)
            ch_1,ch_2,ch_3=rosette_to_tensor(ch_1,ch_2,ch_3)
            gages[3*i]=ch_1
            gages[3*i+1]=ch_2
            gages[3*i+2]=ch_3

        # various useful variables
        avg_before = np.mean(gages[:,:1000],axis=-1)
        avg_after  = np.mean(gages[:,-1000:],axis=-1)
        eps_yy=avg_before[1::3]
        eps_xy=avg_before[2::3]
        mu = eps_xy/eps_yy
        mus.append(mu)
        eps_yys.append(eps_yy)
        eps_xys.append(eps_xy)
        loading_contrast.append( (eps_yy[5] - np.mean(eps_yy[[i for i in range(10) if i!=5]])) /np.mean(eps_yy) )
        fns.append(np.mean(np.abs( fn[:1000] )))
        from_solid.append(int(manip[6:]) in solids)

times=np.array(times)
mus=np.array(mus)
mus[:,5]=np.nan*np.zeros(mus[:,5].shape)
eps_yys=np.array(eps_yys)
eps_xys=np.array(eps_xys)
loading_contrast= np.array(loading_contrast)
fns = np.array(fns)*500/3
from_solid=np.array(from_solid)
# Description of content :
# n : number of events
# m : number of rosettes
# times : (n,m), contains the time of passage of the rupture for each event and rosette
# mus   : (n,m), contains the \mu=\eps_{xy} / \eps_{yy} before crack for each event and rosette
# eps_yys:(n,m), contains the \eps_{yy} before crack for each event and rosette
# loading_contrast : (n,), contains the loading contrast of each event
# fns : (n,), contains the normal force of each event



to_save={}
to_save["times"]=times
to_save["mus"]=mus
to_save["eps_yys"]=eps_yys
to_save["eps_xys"]=eps_xys
to_save["loading_contrast"]=loading_contrast
to_save["fns"]=fns
to_save["from_solid"]=from_solid

np.save(loc_figures+"hist_data.npy",to_save)



## create a histogram of departure zone - max mu, min mu and eps_yy


# Find first time in each
index_sort = np.array([np.argsort(times_j) for times_j in times])
index_start=index_sort[:,0]
index_second=index_sort[:,1]
index_third=index_sort[:,2]
acceptable_location = [0,1,2,3,4,6,7,8,9]
index_test=[]

max_mu = [list(mu).index(max(mu[acceptable_location])) for mu in mus]
min_mu = [list(mu).index(min(mu[acceptable_location])) for mu in mus]
max_yy = [list(yy).index(max(yy[acceptable_location])) for yy in eps_yys]
min_yy = [list(yy).index(min(yy[acceptable_location])) for yy in eps_yys]
max_xy = [list(xy).index(max(xy[acceptable_location])) for xy in eps_xys]
min_xy = [list(xy).index(min(xy[acceptable_location])) for xy in eps_xys]
histogram_start = np.array([list(index_start).count(i) for i in range(10)])
histogram_start_solid = np.array([list(index_start[from_solid]).count(i) for i in range(10)])
histogram_start_gran = np.array([list(index_start[loading_contrast>1.5]).count(i) for i in range(10)])
histogram_minmu =  np.array([min_mu.count(i) for i in range(10)])
histogram_maxmu =  np.array([max_mu.count(i) for i in range(10)])
histogram_maxyy =  np.array([max_yy.count(i) for i in range(10)])
histogram_minyy =  np.array([min_yy.count(i) for i in range(10)])
histogram_maxxy =  np.array([max_xy.count(i) for i in range(10)])
histogram_minxy =  np.array([min_xy.count(i) for i in range(10)])

histogram_test = np.array([0.0 for _ in range(10)])

for i in range(len(times)):

    if times[i][index_second[i]] - times[i][index_start[i]] < 0.3 * (times[i][index_third[i]] - times[i][index_second[i]]) :
        histogram_test[index_start[i]]+=0.5
        histogram_test[index_second[i]]+=0.5
        index_test.append(index_start[i])
        index_test.append(index_second[i])
    else:
        histogram_test[index_start[i]]+=1
        index_test.append(index_start[i])
        index_test.append(index_start[i])



## make the plots : normalized
y_min = -0.05


#n starting point
fig,ax=plt.subplots()
plt.bar(x,histogram_start/histogram_start.sum(),width = 0.8*min(np.diff(x)))
plt.grid(which="both")
plt.suptitle("Jauge de départ du crack")
plt.title("tous chargements confondus")
plt.xlabel("position (cm)")
plt.ylabel("proportion")


temp_nice_plot(y_min,x,ax=ax)

plt.savefig(loc_figures+"start_single_hist_norm.svg")
plt.savefig(loc_figures+"start_single_hist_norm.png",dpi=600)
plt.close('all')


#n starting point solid solid
fig,ax=plt.subplots()
plt.bar(x,histogram_start_solid/histogram_start_solid.sum(),width = 0.8*min(np.diff(x)))
plt.grid(which="both")
plt.suptitle("Jauge de départ du crack")
plt.title("Solide solide")
plt.xlabel("position (cm)")
plt.ylabel("proportion")


temp_nice_plot(y_min,x,ax=ax)

plt.savefig(loc_figures+"start_solid_single_hist_norm.svg")
plt.savefig(loc_figures+"start_solid_single_hist_norm.png",dpi=600)
plt.close('all')



#n starting point gran
fig,ax=plt.subplots()
plt.bar(x,histogram_start_gran/histogram_start_gran.sum(),width = 0.8*min(np.diff(x)))
plt.grid(which="both")
plt.suptitle("Jauge de départ du crack")
plt.title("Granulaire très dense")
plt.xlabel("position (cm)")
plt.ylabel("proportion")


temp_nice_plot(y_min,x,ax=ax)

plt.savefig(loc_figures+"start_gran_single_hist_norm.svg")
plt.savefig(loc_figures+"start_gran_single_hist_norm.png",dpi=600)
plt.close('all')


fig,ax=plt.subplots()
plt.bar(x,histogram_start_solid/histogram_start_solid.sum(),width = 0.8*min(np.diff(x)),alpha = 0.5,label="Solide-Solide")
plt.bar(x,histogram_start_gran/histogram_start_gran.sum(),width = 0.8*min(np.diff(x)),label="Granulaire dense",alpha=0.5)

plt.grid(which="both")
plt.suptitle("Jauge de départ du crack")
#plt.title("Solide solide")
plt.xlabel("position (cm)")
plt.ylabel("proportion")
plt.legend()

temp_nice_plot(y_min,x,ax=ax)

plt.savefig(loc_figures+"start_solidVSgran_single_hist_norm.svg")
plt.savefig(loc_figures+"start_solidVSgran_single_hist_norm.png",dpi=600)
plt.close('all')





# min mu
fig,ax=plt.subplots()
plt.bar(x,histogram_minmu/histogram_minmu.sum(),width = 0.8*min(np.diff(x)))
plt.grid(which="both")
plt.suptitle("Minimum de $\mu$ (trou exclus)")
plt.xlabel("position (cm)")
plt.ylabel("proportion")

temp_nice_plot(y_min,x,ax=ax)
plt.savefig(loc_figures+"minmu_single_hist_norm.svg")
plt.savefig(loc_figures+"minmu_single_hist_norm.png",dpi=600)
plt.close('all')


# max mu
fig,ax=plt.subplots()
plt.bar(x,histogram_maxmu/histogram_maxmu.sum(),width = 0.8*min(np.diff(x)))
plt.grid(which="both")
plt.suptitle("Maximum de $\mu$ (trou exclus)")
plt.xlabel("position (cm)")
plt.ylabel("proportion")

temp_nice_plot(y_min,x,ax=ax)
plt.savefig(loc_figures+"maxmu_single_hist_norm.svg")
plt.savefig(loc_figures+"maxmu_single_hist_norm.png",dpi=600)
plt.close('all')


# max eps yy
fig,ax=plt.subplots()
plt.bar(x,histogram_maxyy/histogram_maxyy.sum(),width = 0.8*min(np.diff(x)))
plt.grid(which="both")
plt.suptitle("Maximum de $\epsilon_{yy}$ (trou exclus)")
plt.xlabel("position (cm)")
plt.ylabel("proportion")

temp_nice_plot(y_min,x,ax=ax)
plt.savefig(loc_figures+"maxyy_single_hist_norm.svg")
plt.savefig(loc_figures+"maxyy_single_hist_norm.png",dpi=600)
plt.close('all')


# min eps yy
fig,ax=plt.subplots()
plt.bar(x,histogram_minyy/histogram_minyy.sum(),width = 0.8*min(np.diff(x)))
plt.grid(which="both")
plt.suptitle("Minimum de $\epsilon_{yy}$ (trou exclus)")
plt.xlabel("position (cm)")
plt.ylabel("proportion")

temp_nice_plot(y_min,x,ax=ax)
plt.savefig(loc_figures+"minyy_single_hist_norm.svg")
plt.savefig(loc_figures+"minyy_single_hist_norm.png",dpi=600)
plt.close('all')



# max eps xy
fig,ax=plt.subplots()
plt.bar(x,histogram_maxxy/histogram_maxxy.sum(),width = 0.8*min(np.diff(x)))
plt.grid(which="both")
plt.suptitle("Maximum de $\epsilon_{xy}$ (trou exclus)")
plt.xlabel("position (cm)")
plt.ylabel("proportion")

temp_nice_plot(y_min,x,ax=ax)
plt.savefig(loc_figures+"maxxy_single_hist_norm.svg")
plt.savefig(loc_figures+"maxxy_single_hist_norm.png",dpi=600)
plt.close('all')


# min eps xy
fig,ax=plt.subplots()
plt.bar(x,histogram_minxy/histogram_minxy.sum(),width = 0.8*min(np.diff(x)))
plt.grid(which="both")
plt.suptitle("Minimum de $\epsilon_{xy}$ (trou exclus)")
plt.xlabel("position (cm)")
plt.ylabel("proportion")

temp_nice_plot(y_min,x,ax=ax)
plt.savefig(loc_figures+"minxy_single_hist_norm.svg")
plt.savefig(loc_figures+"minxy_single_hist_norm.png",dpi=600)
plt.close('all')



## make the plots : not normalized
y_min = -2


#n starting point
fig,ax=plt.subplots()
plt.bar(x,histogram_start,width = 0.8*min(np.diff(x)))
plt.grid(which="both")
plt.suptitle("Jauge de départ du crack")
plt.title("tous chargements confondus")
plt.xlabel("position (cm)")
plt.ylabel("occurrences")


temp_nice_plot(y_min,x,ax=ax)

plt.savefig(loc_figures+"start_single_hist.svg")
plt.savefig(loc_figures+"start_single_hist.png",dpi=600)
plt.close('all')




fig,ax=plt.subplots()
plt.bar(x,histogram_test,width = 0.8*min(np.diff(x)))
plt.grid(which="both")
plt.suptitle("Jauge de départ du crack (alternatif)")
plt.title("tous chargements confondus")
plt.xlabel("position (cm)")
plt.ylabel("occurrences")


temp_nice_plot(y_min,x,ax=ax)

plt.savefig(loc_figures+"start_alt_single_hist.svg")
plt.savefig(loc_figures+"start_alt_single_hist.png",dpi=600)
plt.close('all')




# min mu
fig,ax=plt.subplots()
plt.bar(x,histogram_minmu,width = 0.8*min(np.diff(x)))
plt.grid(which="both")
plt.suptitle("Minimum de $\mu$ (trou exclus)")
plt.xlabel("position (cm)")
plt.ylabel("occurrences")


temp_nice_plot(y_min,x,ax=ax)
plt.savefig(loc_figures+"minmu_single_hist.svg")
plt.savefig(loc_figures+"minmu_single_hist.png",dpi=600)
plt.close('all')


# max mu
fig,ax=plt.subplots()
plt.bar(x,histogram_maxmu,width = 0.8*min(np.diff(x)))
plt.grid(which="both")
plt.suptitle("Maximum de $\mu$ (trou exclus)")
plt.xlabel("position (cm)")
plt.ylabel("occurrences")

temp_nice_plot(y_min,x,ax=ax)
plt.savefig(loc_figures+"maxmu_single_hist.svg")
plt.savefig(loc_figures+"maxmu_single_hist.png",dpi=600)
plt.close('all')


# max eps yy
fig,ax=plt.subplots()
plt.bar(x,histogram_maxyy,width = 0.8*min(np.diff(x)))
plt.grid(which="both")
plt.suptitle("Maximum de $\epsilon_{yy}$ (trou exclus)")
plt.xlabel("position (cm)")
plt.ylabel("occurrences")

temp_nice_plot(y_min,x,ax=ax)
plt.savefig(loc_figures+"maxyy_single_hist.svg")
plt.savefig(loc_figures+"maxyy_single_hist.png",dpi=600)
plt.close('all')



# min eps yy
fig,ax=plt.subplots()
plt.bar(x,histogram_minyy,width = 0.8*min(np.diff(x)))
plt.grid(which="both")
plt.suptitle("Minimum de $\epsilon_{yy}$ (trou exclus)")
plt.xlabel("position (cm)")
plt.ylabel("occurrences")

temp_nice_plot(y_min,x,ax=ax)
plt.savefig(loc_figures+"minyy_single_hist.svg")
plt.savefig(loc_figures+"minyy_single_hist.png",dpi=600)
plt.close('all')


# max eps xy
fig,ax=plt.subplots()
plt.bar(x,histogram_maxxy,width = 0.8*min(np.diff(x)))
plt.grid(which="both")
plt.suptitle("Maximum de $\epsilon_{xy}$ (trou exclus)")
plt.xlabel("position (cm)")
plt.ylabel("occurrences")

temp_nice_plot(y_min,x,ax=ax)
plt.savefig(loc_figures+"maxxy_single_hist.svg")
plt.savefig(loc_figures+"maxxy_single_hist.png",dpi=600)
plt.close('all')



# min eps xy
fig,ax=plt.subplots()
plt.bar(x,histogram_minxy,width = 0.8*min(np.diff(x)))
plt.grid(which="both")
plt.suptitle("Minimum de $\epsilon_{xy}$ (trou exclus)")
plt.xlabel("position (cm)")
plt.ylabel("occurrences")

temp_nice_plot(y_min,x,ax=ax)
plt.savefig(loc_figures+"minxy_single_hist.svg")
plt.savefig(loc_figures+"minxy_single_hist.png",dpi=600)
plt.close('all')






### Binned histogram




def plot_quad_hist(bin_width, bin_variable, x_axis, to_count, normalized, suptitle, bin_var_name="LC", save_name=False):
    """
    temporary function to simplify the plotting of the quad hist
    """
    # define the bins
    max_lc = np.ceil(max(bin_variable)/bin_width)*bin_width
    min_lc = np.floor(min(bin_variable)/bin_width)*bin_width
    n_bin =  int(np.ceil((max_lc-min_lc)/bin_width))

    # sort the values in the bins
    hists = []
    for i in range(n_bin):
        hists.append([([to_count[k]
                            for k in range(len(to_count))
                            if  bin_variable[k]<min_lc+bin_width*(i+1)
                            and bin_variable[k]>=min_lc+bin_width*i
                        ]).count(j) for j in range(10)])

    hists=np.array(hists)

    # Plot the result, normalized
    nrows=int(np.ceil(n_bin/2))
    fig, axes = plt.subplots(nrows=nrows, ncols=2, sharey=True)
    fig.set_size_inches(8,nrows*2)
    fig.subplots_adjust(hspace=0.5)

    for i, ax in enumerate(axes.flat):
        try :
            if normalized:
                y_min=-0.05
                ax.bar(x_axis, hists[i]/hists[i].sum(),width=0.8*min(np.diff(x_axis)))
            else :
                y_min=-2
                ax.bar(x_axis, hists[i],width=0.8*min(np.diff(x_axis)))
            ax.set_title("{:.1f} $\leq$ {} < {:.1f}".format(min_lc+i*bin_width,bin_var_name,min_lc+(i+1)*bin_width))
            temp_nice_plot(y_min,x_axis,ax=ax)
            ax.grid(which="both")

        except:
            ax.axis('off')

    plt.suptitle(suptitle)
    if normalized :
        axes[0][0].set_ylabel("Proportion (pour chaque bin)",fontsize="5")
        if save_name:
            plt.savefig(loc_figures+save_name+"_quad_hist_norm.svg")
            plt.savefig(loc_figures+save_name+"_quad_hist_norm.png",dpi=600)
        else:
            plt.show()
    else :
        axes[0][0].set_ylabel("Occurences (pour chaque bin)",fontsize="5")
        if save_name:
            plt.savefig(loc_figures+save_name+"_quad_hist.svg")
            plt.savefig(loc_figures+save_name+"_quad_hist.png",dpi=600)
        else:
            plt.show()
    plt.close('all')




to_counts = [min_mu,max_mu,min_xy,max_xy,min_yy,max_yy,index_start]
suptitles = [["in","$\mu$"],["ax","$\mu$"],["in","$\epsilon_{xy}$"],["ax","$\epsilon_{xy}$"],["in","$\epsilon_{yy}$"],["ax","$\epsilon_{yy}$"]]
suptitles = ["M{}imum de {} en fonction du Loading Contrast".format(sup[0],sup[1]) for sup in suptitles] + ["Jauge de départ en fonction du Loading Contrast"]
save_names = ["minmu","maxmu","minxy","maxxy","minyy","maxyy","start"]

for i in range(len(to_counts)):
    for normalized in [True,False]:
        plot_quad_hist(bin_width    = 0.5,
                       bin_variable = loading_contrast,
                       x_axis       = x,
                       to_count     = to_counts[i],
                       normalized   = normalized,
                       suptitle     = suptitles[i],
                       bin_var_name = "LC",
                       save_name    = save_names[i])


for normalized in [True,False]:
    plot_quad_hist(bin_width    = 0.5,
                       bin_variable = np.reshape([[i,i] for i in loading_contrast],len(loading_contrast)*2),
                       x_axis       = x,
                       to_count     = index_test,
                       normalized   = normalized,
                       suptitle     = "Jauge de départ en fonction du Loading Contrast (alt)",
                       bin_var_name = "LC",
                       save_name    = "start_alt")


plot_quad_hist(bin_width    = 10,
                bin_variable = fns,
                x_axis       = x,
                to_count     = index_start,
                normalized   = True,
                suptitle     = "Jauge de depart en fonction de Fn",
                bin_var_name = "Fn",
                save_name    = "start_Fn")


plot_quad_hist(bin_width    = 10,
                bin_variable = fns,
                x_axis       = x,
                to_count     = index_start,
                normalized   = False,
                suptitle     = "Jauge de depart en fonction de Fn",
                bin_var_name = "Fn",
                save_name    = "start_Fn")


















