# plot_figures.py

import numpy as np
import matplotlib.pyplot as plt

def plot_distribution_data(data):

    fig, axs = plt.subplots(figsize=(4,6),ncols=1, nrows=2)

    attentions = list(data.keys())
    
    
    n_trials,n_inputs, n_neurons = data[attentions[0]]["S"].shape


    bins = np.floor(np.sqrt(n_trials)).astype(int)

    for i in range(len(attentions)):
        axs[i].hist(data[attentions[i]]["S"][:,0,0].ravel(), bins=np.linspace(0,25,bins),alpha=0.6)
        for j in range(n_inputs):
            axs[i].hist(data[attentions[i]]["R"][:,j,0].ravel(), bins=np.linspace(0,25,bins),alpha=0.6)


        axs[i].set_xlabel("Firing Rate (Hz)")
    axs[0].text(0.1,0.9,"State A",horizontalalignment='left',rotation=0,color="b", transform=axs[0].transAxes,weight="bold")
    axs[1].text(0.1,0.9,"State B",horizontalalignment='left',rotation=0,color="r", transform=axs[1].transAxes,weight="bold")


    ylimin,ylimax = np.inf,-np.inf
    for i in range(2):
        ylim = axs[i].get_ylim()
        ylimin = min(ylim[0],ylimin)
        ylimax = max(ylim[1],ylimax)
    for i in range(2):
        axs[i].set_ylim(ylimin,ylimax)

    ylimin,ylimax = np.inf,-np.inf
    for i in range(2):
        ylim = axs[i].get_xlim()
        ylimin = min(ylim[0],ylimin)
        ylimax = max(ylim[1],ylimax)
    for i in range(2):
        axs[i].set_xlim(ylimin,ylimax)


    for i in range(2):
        axs[i].set_yticks([0,100,200])
        axs[i].set_yticklabels([0,10,20])
        axs[i].set_ylabel("Occurrence (%)")


    axs[0].legend(["Spontaneous","Stimulus #1","Stimulus #2","Stimulus #3","Stimulus #4"],ncol=2,loc='upper center',bbox_to_anchor=(0.5,1.4))
    
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.4)



def plot_information_content(data):
    
    i_detection = data[0]
    i_detection_interval = data[1]
    i_differentiation = data[2]
    i_differentiation_interval = data[3]
    
    fig, axs = plt.subplots(figsize=(8,3),ncols=2, nrows=1)

    colors = ["b","r"]
    attentions = len(data[0])
    for i in range(attentions):
        axs[0].bar(np.array([1,2,3,4])- (-1)**i*0.13,i_detection[i],yerr=i_detection_interval[i], width=0.2,color=colors[i])

    axs[0].set_title('Information Detection')
    axs[0].set_xticks([1,2,3,4])
    axs[0].set_xticklabels(["#1","#2","#3","#4"])
    axs[0].set_ylabel("NMI")
    axs[0].set_xlabel("Stimulus")
    axs[0].legend(["State A","State B"])




    for i in range(attentions):
        axs[1].bar(np.array([1])- (-1)**i*0.13,i_differentiation[i],yerr=i_differentiation_interval[i], width=0.2,color=colors[i])
    axs[1].set_xlim(0.5,1.5)
    axs[1].set_title('Information Differentiation')
    axs[1].set_xticks([0.87,1.13])
    axs[1].set_xticklabels(["State A","State B"])
    axs[1].set_ylabel("NMI")



    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)


