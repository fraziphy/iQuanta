# generate_raw_data.py
import numpy as np
import scripts.config

def Generate_Spike_Times(firing_rate,duration):
    '''
    Generating spike times of a neuron with a Poisson spike train
    '''
  
  
    num_spikes = np.floor(firing_rate * duration).astype(int) # The expected number of spikes in the specified duration.

    spike_times = np.random.exponential(1 / firing_rate, size=num_spikes) # The random spike times according to an exponential distribution with a mean inter-spike interval corresponding to the inverse of the firing rate.
    spike_times = np.cumsum(spike_times) # Convert the spike times into absolute spike times
    spike_times = spike_times[spike_times < duration] # Filters out any spike times that occur after the specified duration

    return spike_times



def Generate_Signle_Trial(spon_activity,evok_activity):

    spike_trains_1 = Generate_Spike_Times(spon_activity,scripts.config.pre_stimulus_time) # Prestimulus spike trains
    spike_trains_2 = Generate_Spike_Times(evok_activity,scripts.config.stimulus_duration) # Stimulus-evoked spike trains
    spike_trains_2 = spike_trains_2 + scripts.config.pre_stimulus_time # To time lock the spike trains
    spike_trains_3 = Generate_Spike_Times(spon_activity,scripts.config.recording_time - (scripts.config.pre_stimulus_time+scripts.config.stimulus_duration)) # Stimulus-evoked spike trains
    spike_trains_3 = spike_trains_3 + (scripts.config.pre_stimulus_time+scripts.config.stimulus_duration) # To time lock the spike trains

    # Append spike times
    spike_trains = np.append(spike_trains_1,spike_trains_2)
    spike_trains = np.append(spike_trains,spike_trains_3)
    
    return spike_trains - scripts.config.pre_stimulus_time

    
    
def Generate_Raw_Spiking_Data(n_neurons,n_inputs,n_trials,attentions):
    
    # Empty arrays to store the firing rates for generating spike trains
    spon_activity = np.zeros((n_trials,n_inputs,n_neurons),dtype=float)
    evok_activity = np.zeros((n_trials,n_inputs,n_neurons),dtype=float)
    
    
    
    # The random values for firing rates
    for i in range(n_neurons):
        spon_activity[:,:,i] = np.random.normal(scripts.config.mean_rate, scripts.config.std_rate, (n_trials,n_inputs))
        for j in range(n_inputs):
            evok_activity[:,j,i] = np.random.normal(scripts.config.mean_rate, scripts.config.std_rate, n_trials) + (j+scripts.config.response_amp_spiking)
    
    spon_activity[spon_activity<=0] = 0.5
    evok_activity[evok_activity<=0] = 0.5
    
    
    
    data = {}
    for ii,attention in enumerate(attentions):
        raw_data = [[] for _ in range(n_neurons)]
        for k in range(n_neurons):
            raw_data_neuron = [[] for _ in range(n_inputs)]
            for j in range(n_inputs):
                # Generate Poisson spike train data
                raw_data_neuron_input = [[] for _ in range(n_trials)]
                for i in range(n_trials):
                    raw_data_neuron_input[i] = Generate_Signle_Trial(spon_activity[i,j,k],evok_activity[i,j,k]+attention)
                raw_data_neuron[j] = raw_data_neuron_input
            raw_data[k] = raw_data_neuron
        data[attention] = raw_data

    return data





def Generate_Distribution_Data(n_neurons,n_inputs,n_trials,attentions):
    
    data_plot_1 = {}

    for attention in attentions:

        data_plot_1["attention={}".format(attention)] = {}
        
        bins = np.floor(np.sqrt(n_trials)).astype(int)

        S = np.random.normal(scripts.config.spon_dist_mean,scripts.config.spon_dist_std,(n_trials,n_inputs,n_neurons))
        R = np.zeros((n_trials,n_inputs,n_neurons),dtype=float)
        
        for i in range(n_neurons):
            for j in range(n_inputs):
                R[:,j,i] = np.random.normal(scripts.config.spon_dist_mean+(j+1) * scripts.config.response_amp+attention,scripts.config.evok_dist_std,(n_trials))
                
        
        data_plot_1["attention={}".format(attention)]["R"] = R
        data_plot_1["attention={}".format(attention)]["S"] = S
            
    return data_plot_1

