# process_data.py
import numpy as np
from iQuanta.custom_funcs import Information_Content as iquanta
import scripts.config

def Pre_Process_Data(data):

    stimulus_offset = 1 # OStimuli offset

    n_neurons = len(data)
    n_inputs = len(data[0])
    n_trials = len(data[0][0])
    
    spon_data = np.empty((n_trials,n_inputs,n_neurons),dtype=float)
    evok_data = np.empty((n_trials,n_inputs,n_neurons),dtype=float)
    
    for k in range(n_neurons):
        for j in range(n_inputs):
            for i in range(n_trials):
                x = data[k][j][i]
                spon_data[i,j,k] = len(x[x<0]) # The number of spike prior to stimulus onset. Given that The recording time in prestimulus interval is 1 seconds, the umber of spikes determines the firing rate
                evok_data[i,j,k] = len(x[(x>=0) & (x<scripts.config.recording_time)])

    return spon_data/scripts.config.pre_stimulus_time, evok_data/scripts.config.recording_time




def Process_Data_Ditribution(data, method="K_means_clustering"):

    i_detection = []
    i_detection_interval = []

    i_differentiation = []
    i_differentiation_interval = []
    
    attentions = list(data.keys())

    for attention in attentions:

        spontaneous_activities = data[attention]["S"]
        evoked_responses = data[attention]["R"]

        i_detec,i_diff = iquanta(spontaneous_activities,evoked_responses,method)

        i_detection.append(i_detec[0])
        i_detection_interval.append(i_detec[1])

        i_differentiation.append(i_diff[0])
        i_differentiation_interval.append(i_diff[1])


    return i_detection, i_detection_interval, i_differentiation, i_differentiation_interval
