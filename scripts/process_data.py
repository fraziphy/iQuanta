# process_data.py
import numpy as np
from sklearn.feature_selection import mutual_info_classif
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



def Mutual_Information(X,Y):
    n_bins = np.floor(np.sqrt(len(Y))).astype(int)
    n_feature = X.shape[1]
    bins_seq = [np.linspace(0, 60, n_bins) for i in range(n_feature)] # Bin edges for each dimension
    # The joint probability of X
    r = np.histogramdd(X, bins=bins_seq)
    p_R = r[0]/r[0].sum()
    p_R_ravel = p_R.ravel()

    # The entropy of X
    H_R = - (p_R_ravel[p_R_ravel!=0] * np.log2(p_R_ravel[p_R_ravel!=0])).sum()


    # The conditional probability of X
    n_inputs = len(np.unique(Y))
    input_count = np.empty(n_inputs,dtype=float)
    p_R_S = {}
    for i in range(n_inputs):
        input_count[i] = len(Y[Y==i])
        x = X[Y==i]
        r = np.histogramdd(x, bins=bins_seq)
        p_R_s_aux = r[0]/r[0].sum()
        p_R_S[i] = p_R_s_aux.ravel()
    p_S = input_count/input_count.sum()

    # The entropy of X given Y
    H_R_S = [- p_S[i] * (p_R_S[i][p_R_S[i]!=0] * np.log2(p_R_S[i][p_R_S[i]!=0])).sum() for i in range(n_inputs)]
    H_R_S = np.array(H_R_S)
    H_R_S = H_R_S.sum()

    # Mutual Information
    I = H_R - H_R_S

    return I


def Stratified_K_FOLD_MI(data,cond,k_fold=10):
    
    # The function is implemented specifically for the stratified k-fold sampling algorithm. It outputs the mutual information between the provided arrays
    n_trial,n_inputs,n_feature = data.shape 
    
    # Preprocess the data
    # X is a columnar array representing the data. The first n_trial elements correspond to the neural activity in response to the first input, the next set to the second input, and so forth. Note that X has to have the column dimention equal to the nmber of data features. Use .reshape(-1, 1) if your data has a single feature.
    X = data.ravel().reshape(-1,n_feature) 
    
    # Generate a unique identifier (the target variables) for each input across trials.
    y = [i for i in range(n_inputs)]*n_trial
    y = np.array(y)
    
    # Shuffle the data, X, and the target variables, y, across trials and inputs.
    p = np.random.permutation(len(y)) 
    X = X[p] 
    y = y[p] 
    
    # The function to provides train/test indices to split data into k folds
    skf = StratifiedKFold(n_splits=k_fold,random_state=1,shuffle=True)
    skf.get_n_splits(X, y)
    
    # Empty arrays to store the MI on each fold.
    MI = np.zeros(k_fold,dtype=float) 
    
    # Loop over the k_folds
    for i, (train_index, test_index) in enumerate (skf.split(X, y)): # train_index and test_index represent the indices for the train and test sets, respectively. Training sets are used to compute the MI
        
        # Split the data and the target variables according to the traning and test indices
        X_train,X_test = X[train_index],X[test_index]
        y_train,y_test = y[train_index],y[test_index] 
        
        # Compute and store the MI on the training set
        if cond=="distribution_data":
            MI[i] = Mutual_Information(X_train,y_train)
        else:
            MI[i] = mutual_info_classif(X_train,y_train)[0]
    
    return MI.mean(),MI.std()
