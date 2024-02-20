# custom_funcs.py
import iQuanta.config as config
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.cluster import normalized_mutual_info_score
# 
# 
# 
# 
#
# Define the function to extract information content in neural data.
def Information_Content(spontaneous_data,evoked_data,method="k_means_clustering"):
    
    algorithm = K_Means_Clustering
    if method == "knn":
        algorithm = KNN
    if method == "neural_network":
        algorithm = Neural_Network
        
    # Spontaneous and evoked data are arranged into three dimensions (n_trials * n_inputs * n_features), corresponding to the number of trials, inputs, and features. The number of features can represent various neural characteristics. For instance, it can denote the number of neurons. If there is only one feature, the data should be in the format of (n_trials * n_inputs * 1).
    
    t_crit = -stats.t.ppf((100-config.confidence_interval)/100/2,config.k_fold) # The t-critical value is used for determining the boundaries of the corresponding confidence interval in statistical analysis.
    
    # Dimensions of the data.
    n_trial,n_inputs,n_feature = evoked_data.shape
    
    
    ################## INFORMATION DETECTION
    info_detec = np.zeros((2,n_inputs),dtype=float) # An empty arrays is initialized to store the values of information detection on test sets for each corresponding input. The second row stores the values for the 95% confidence interval.
    
    # Iterates over the inputs to quantify information detection for each corresponding input. The algorithm used to quantify information detection is determined by the method specified, which can be either supervised or unsupervised.
    for i in range(n_inputs):
        
        X = np.zeros((n_trial,2,n_feature),dtype=float) # An auxiliary array is initialized to host the spontaneous and evoked data together.
        X[:,0] = spontaneous_data[:,i]
        X[:,1] = evoked_data[:,i]
        
        train, test = Stratified_K_FOLD(X, algorithm,k_fold=config.k_fold) # The function performing the stratified K-fold sampling on data. It returns the output of the trained "algorithm" on the training and test sets.
        
        info_detec[0,i] = test.mean() # The mean over k-fold sampling to quantify information detection on the test set.
        info_detec[1,i] = t_crit * test.std() / np.sqrt(config.k_fold) # 95% confidence interval.
            
        return info_detec
    
    
    ################## INFORMATION DIFFERENTIATION
    info_diff = np.zeros(2,dtype=float) # An empty array is initialized to store the values of information differentiation on test sets. The second value stores the 95% confidence interval.
    
    train, test = Stratified_K_FOLD(evoked_data, algorithm,k_fold=config.k_fold) # The function performing the stratified K-fold sampling on data. It returns the output of the "algorithm" on the provided "X".

    info_diff[0] = test.mean() # The mean over k-fold sampling to quantify information detection on the test set.
    info_diff[1] = t_crit * test.std() / np.sqrt(config.k_fold) # 95% confidence interval.
            
    return info_detec, info_diff
# 
# 
# 
# 
#
# Define the function performing the stratified K-fold sampling on the accuracy of the corresponding classifier on the data
def Stratified_K_FOLD(data, Classifier_Algorithm,k_fold):
    
    # The function is implemented specifically for the stratified k-fold sampling algorithm. It outputs the accuracy attained by the employed classification algorithm across the k-folds of both the training and testing data sets.
    
    # rng serves as the pseudo-random-number generator used to randomly shuffle the data across trials and inputs.
    # data corresponds to neural responses to inputs, organized in a matrix with dimensions n_trial * n_inputs.
    # For assessing information detection, the first column denotes spontaneous neural activity, while the second column represents neural responses to a given input.
    # To evaluate information differentiation, each column illustrates neural responses to different inputs.
    # Classifier_Algorithm refers to the classification algorithm employed—utilizing both K-means clustering and logistic classification—to measure information content in neural signals.
    # k_fold represents the number of subsets (folds) for the stratified k-fold sampling algorithm.
    

    # n_trial: the number of trials. n_inputs: the number of inputs. Note that n_input = 2 for information detection (for spontaneous and evoked neural responses to a given input) and n_input = number_of_inputs for information differentiation.
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
    
    # The function to provides train/test indices to split data in train/test sets. For details, please refer to https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
    skf = StratifiedKFold(n_splits=config.k_fold,random_state=1,shuffle=True)
    skf.get_n_splits(X, y)
    
    # Empty arrays to store the accuracy scores on the training and test sets.
    accuracy_train = np.zeros(config.k_fold,dtype=float) 
    accuracy_test = np.zeros(config.k_fold,dtype=float) 
    
    # The function to standardize data across features
    scaler = StandardScaler()
    
    # Loop over the k_folds
    for i, (train_index, test_index) in enumerate (skf.split(X, y)): # train_index and test_index represent the indices for the train and test sets, respectively.
        
        # Split the data and the target variables according to the traning and test indices
        X_train,X_test = X[train_index],X[test_index]
        y_train,y_test = y[train_index],y[test_index] 
        
        # Standardize features
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Compute the accuracy of the given classifier 
        classifier_accuracy_train, classifier_accuracy_test = Classifier_Algorithm(X_train, y_train, X_test, y_test) 

        # Store the accuracy scores on the training and test sets.
        accuracy_train[i] = classifier_accuracy_train 
        accuracy_test[i] = classifier_accuracy_test 
    
    return accuracy_train, accuracy_test
# 
# 
# 
# 
#
# Define the class describing the neural network with one hidden layer for multinomial classification.
class NEURAL_NETWORK_CLASS(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__() # Call the  superclass's methods (nn).
        fc1 = nn.Linear(input_dim, int(2/3*input_dim)+output_dim) # The input layer. Change the number of hidden neurons acording to your need. The number of hidden neurons should be less than twice the size of the input layer.
        relu = nn.ReLU() # The hidden layer.
        fc2 = nn.Linear(int(2/3*input_dim)+output_dim, output_dim) # The output layer.
    
    def forward(self, x):
        x = fc1(x) # Output of the input layer.
        x = relu(x) # Output of the hidden layer.
        x = fc2(x) # Output of the output layer.
        return x
# 
# 
# 
# 
#
# Define the function calculating the accuracy of the neural network classifier on the data
def Neural_Network(data_train, label_train, data_test, label_test):
    
    # Convert the training and test data and the training target variable to PyTorch tensors
    data_train_tensor = torch.tensor(data_train, dtype=torch.float32)
    label_train_tensor = torch.tensor(label_train, dtype=torch.long)
    data_test_tensor = torch.tensor(data_test, dtype=torch.float32)
    
    feature_dim = data_train_tensor.shape[1]
    class_dim = len(torch.unique(label_train_tensor))
    
    ################## MODEL SELECTION 
    model = NEURAL_NETWORK_CLASS(feature_dim, class_dim)
    
    ################## ESTIMATION OF MODEL PARAMETERS
    # Define the loss function and the optimizer along with the learning rate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # The Adam optimizer, short for “Adaptive Moment Estimation
    
    # Train the model on the training set
    for epoch in range(num_epochs):
        
        # Forward pass
        outputs = model(data_train_tensor)
        loss = criterion(outputs, label_train_tensor)
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    ################## PREDICTION
    # Evaluate the model performance
    with torch.no_grad(): 
        model.eval()
        
        label_pred_train_tensor = model(data_train_tensor)
        _, label_pred_train_tensor = torch.max(label_pred_train_tensor, 1)
        label_pred_train = label_pred_train_tensor.numpy()
        
        label_pred_test_tensor = model(data_test_tensor)
        _, label_pred_test_tensor = torch.max(label_pred_test_tensor, 1)
        label_pred_test = label_pred_test_tensor.numpy()
        
    # Calculate the accuracy on the training and the test set
    accuracy_train = accuracy_score(label_train_tensor.numpy(), label_pred_train)
    accuracy_test = accuracy_score(label_test, label_pred_test)
    
    return accuracy_train, accuracy_test
# 
# 
# 
# 
#
# Define the function calculating the accuracy of the K-means clustering classifier on the data
def K_Means_Clustering(data_train, label_train, data_test, label_test):
    
    ################## IDENTIFYING THE NUMBER OF CLUSTERS
    n_clusters = len(np.unique(label_train))
    
    ################## ESTIMATING THE CLUSTER CENTROIDS
    model = KMeans(init="random",n_clusters=n_clusters,n_init=config.n_init,max_iter=config.max_iter)
    cluster_centers = model.fit(data_train).cluster_centers_
    
    ################## PREDICTION
    # Evaluate the model performance
    label_pred_train = model.fit(data_train).labels_
    label_pred_test = model.predict(data_test)
    
    # Calculate the accuracy on the training and the test set. The method used is the normalized mutual information. Details can be found here: https://www.jmlr.org/papers/volume3/strehl02a/strehl02a.pdf
    accuracy_train = normalized_mutual_info_score(label_train, label_pred_train,average_method=config.average_method)
    accuracy_test = normalized_mutual_info_score(label_test, label_pred_test,average_method=config.average_method)
    
    return accuracy_train, accuracy_test
# 
# 
# 
# 
#
# Define the function calculating the accuracy of the KNN classifier on the data
def KNN(data_train, label_train, data_test, label_test):
    
    ################## IDENTIFYING THE NUMBER OF CLUSTERS
    n_clusters = len(np.unique(label_train))
    
    ################## ESTIMATING THE CLUSTER CENTROIDS
    model = KNeighborsClassifier(n_neighbors=np.floor(np.sqrt(len(label_train))).astype(int))
    model.fit(data_train, label_train) 
    
    ################## PREDICTION
    # Evaluate the model performance
    label_pred_train = model.predict(data_train)
    label_pred_test = model.predict(data_test)
    
    # Calculate the accuracy on the training and the test set
    accuracy_train = accuracy_score(label_train, label_pred_train)
    accuracy_test = accuracy_score(label_test, label_pred_test)
    
    return accuracy_train, accuracy_test
