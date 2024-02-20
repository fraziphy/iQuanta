# config.py

from pathlib import Path  # pathlib is seriously awesome!

data_dir = Path('../data/')
data_raw_path = data_dir / 'raw/my_file.pkl'
data_processed_path = data_dir / 'processed/my_file.pkl'

# Stratified_K_FOLD
k_fold = 10 # The number of random samplings for the stratified k-fold sampling algorithm.
confidence_interval = 95 # Here is 95% confidence interval.

# Neural_Network
num_epochs = 1000  # Number of itteration for learning
learning_rate = 0.001 # The learning rate of the stochastic gradient descent

# K_Means_Clustering
max_iter=300 # Maximum number of iterations of the k-means algorithm for a single run
n_init = 10 # Number of times the k-means algorithm is run with different centroid seeds.


# normalized_mutual_info_score
average_method="geometric" # How to compute the normalizer in the denominator.
