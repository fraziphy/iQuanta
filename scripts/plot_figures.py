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


    axs[0].text(-0.2,1.13,"a",horizontalalignment='left', transform=axs[0].transAxes,weight="bold")
    axs[1].text(-0.2,1.13,"b",horizontalalignment='left', transform=axs[1].transAxes,weight="bold")
    
    
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

    
    axs[0].text(-0.2,1.13,"a",horizontalalignment='left', transform=axs[0].transAxes,weight="bold")
    axs[1].text(-0.2,1.13,"b",horizontalalignment='left', transform=axs[1].transAxes,weight="bold")

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)



def plot_spiking_data(data):
    
    attentions = list(data.keys())
    
    n_trials = len(data[attentions[0]][0][0])
    n_inputs = len(data[attentions[0]][0])
    n_neurons = len(data[attentions[0]])
    
    
    
    
    
    neuron_id = 0
    
    fig, ax = plt.subplots(5, n_inputs,figsize=(11, 7))
    color = ["b","r"]
    
    
    ylimmin,ylimmax = np.inf,-np.inf
    
    for kk in range(2):
        for j in range(n_inputs):
            for i in range(20):
                ax[3*kk+0,j].eventplot(data[attentions[kk]][neuron_id][j][i], lineoffsets=i, colors='black', linewidths=1, zorder=2)
            
            
            ax[3*kk+0,j].axis('off')
            ax[3*kk+0,j].set_ylabel('Trials')
            ax[0,j].set_title('Stimulus #{}'.format(j+1))
        
            ax[3*kk+0,j].fill_between([0,1],[-1]*2,[10]*2,color="gray",alpha=0.3, linewidth=0, zorder=1)
                
            
            h = np.array([])
            for i in data[attentions[kk]][neuron_id][j]:
                h = np.append(h,i)
            ax[3*kk+1,j].hist(h,bins=30,color=color[kk],edgecolor='black', linewidth=1.2, zorder=2)
            ax[3*kk+1,j].set_xlabel('Time (s)')
            
            ax[3*kk+1,j].set_xticks([-1,0,1,2])
        
            ylim = ax[3*kk+1,j].get_ylim()
            ylimmin = min(ylim[0],ylimmin)
            ylimmax = max(ylim[1],ylimmax)
        
        
        
        ax[3*kk+1,0].set_ylabel("Firing Rate (Hz)")
    for kk in range(2):
        for j in range(n_inputs):
            ax[3*kk+1,j].set_ylim(ylimmin,ylimmax)
    for kk in range(2):
        for j in range(n_inputs):
            # yticks = ax[3*kk+1,j].get_yticks()
            # ax[3*kk+1,j].set_yticks(yticks)
            # ax[3*kk+1,j].set_yticklabels(yticks/10)
            ax[3*kk+1,j].set_yticks([0,500,1000])
            ax[3*kk+1,j].set_yticklabels([0,50,100])
            
    for kk in range(2):
        for j in range(n_inputs):
            ax[3*kk+1,j].fill_between([0,1],[ylimmin]*2,[ylimmax]*2,color="gray",alpha=0.3, linewidth=0, zorder=1)

    
    ax[1,0].text(-0.4, 1, "State A", transform=ax[1,0].transAxes,rotation=90)
    ax[4,0].text(-0.4, 1, "State B", transform=ax[4,0].transAxes,rotation=90)
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
    
    ax[0,0].text(-0.4,1.13,"a",horizontalalignment='left', transform=ax[0,0].transAxes,weight="bold")
    ax[3,0].text(-0.4,1.13,"b",horizontalalignment='left', transform=ax[3,0].transAxes,weight="bold")

    for i in range(4):
        ax[2,i].remove()
