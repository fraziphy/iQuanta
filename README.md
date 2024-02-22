# iQuanta

## Description

### iQuanta is a Python module developed to formulate a novel framework for quantifying the information content in neural data using machine learning techniques.

One of the primary methods neurons employ to communicate is through the generation of action potentials, also known as spikes. These neural spikes can occur spontaneously or in response to external stimuli, carrying information about the stimuli they encode. However, accurately quantifying the information content within neural response patterns remains a challenging endeavor.

The _**iQuanta**_ module is tailored for the quantification of information content in neural signals. The framework's methodology for quantifying information content is outlined in doi:10.1101/2023.12.04.569905. In summary, it introduces two measures: information detection and information differentiation. Information detection evaluates whether neural responses to a stimulus significantly differ from spontaneous neural activities. Conversely, information differentiation assesses whether neural responses to distinct stimuli significantly vary from one another.

The _**iQuanta**_ module employs a range of supervised and unsupervised machine learning techniques, such as K-means clustering, K-nearest neighbors, and logistic classification algorithms. Information detection gauges how effectively the machine learning algorithm segregates evoked neural responses to a given stimulus from spontaneous neural activities. Similarly, information differentiation measures the algorithm's ability to distinguish evoked neural responses to different stimuli from each other.

In the unsupervised approach, the clustering algorithm's performance in predicting cluster labels was assessed using Normalized Mutual Information (NMI) as an external validation metric. NMI evaluates the resemblance between the predicted and true cluster labels, with reference to available true labels (see Strehl, A., & Ghosh, J. (2002). Cluster ensembles---a knowledge reuse framework for combining multiple partitions. _Journal of machine learning research, 3_(Dec), 583-617.). A score of zero indicates no similarity, while a score of one represents complete similarity between the predicted and true labels. In the supervised setting, the effectiveness of the clustering algorithm in predicting cluster labels was evaluated using accuracy scores.

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## Installation

To install the _**iQuanta**_, ensure the necessary Python libraries **numpy**, **torch**, **sklearn**, and **scipy** are installed. Please verify that the corresponding dependencies are present. To confirm that the scripts and required libraries are installed on your local machine, navigate to the directory of the iQuanta module. You can begin by testing the Python scripts within the project and scripts directories. To do this, install the Python library nose2 and execute it from the command line:
```
$ pip install nose2
$ python -m nose2
```

To **install** the _**iQuanta**_ module from GitHub, run the following command:
```
!pip install git+ssh://git@github.com/fraziphy/iQuanta.git
```
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

To **uninstall** the module, please copy and execute the following command in a single cell:

```
!python -m pip uninstall iQuanta --yes
```

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## Usage

To import the module, use the following syntax:
```
from iQuanta.funcs import iquanta # The Module to quantify information content
```
To quantify information content in neural signals, the  _**iQuanta**_  module accepts three arguments:
```
iquanta(spontaneous_activities,evoked_responses,algorithm)
```
Here, "spontaneous_activities" and "evoked_responses" represent the neural activity in pre-stimulus and post-stimulus intervals, respectively. The "algorithm" parameter refers to the machine learning technique employed for quantifying information content.

Provide your neural data (including both spontaneous_activities and evoked_responses) as a matrix with the following dimensional configuration:

```
number_trials, number_stimuli, number_features
```

The first and second dimensions correspond to the neural signal in each trial for each stimulus. The third dimension signifies different features of the neural signal. For example, it could represent the firing rate of various neurons in single unit recordings, the frequency power of LFP signals, or characteristics like latency and slope of EEG deflections at different electrodes.

To specify the desired algorithm, use the following argument for each algorithm:

    1. Use "K_means_clustering" for K-means clustering algorithm.
    2. Use "knn" for K-nearest neighbors.
    3. Use "neural_network" for logistic classification algorithms.

The _**iquanta**_ function returns the values of information detection and information differentiation in the following order:

```
information_detection, information_differentiation = iquanta(spontaneous_activity,evoked_responses,algorithm)
```
The "information_detection" is a matrix of size (2, number_stimuli). The first and second rows in "information_detection" represent the mean and 95% confidence interval for information detection obtained from the stratified k-fold sampling algorithm, where k is set to 10 (to modify k, adjust the value in the iQuanta.config). Each column represents the values corresponding to the respective stimuli.

Similarly, the first and second entries in "information_differentiation" show the mean and 95% confidence interval for information differentiation derived from the stratified k-fold sampling algorithm (with k set to 10).

As an example where number_stimuli=4:
```
i_detection,i_differentiation = iquanta(spontaneous_activities,evoked_responses,"K_means_clustering")

i_detection:
                    array([[0.29146534, 0.51875989, 0.73027708, 0.89519773],
                            [0.03697001, 0.04568368, 0.03838708, 0.02594112]])

i_differentiation:
                    array([0.26832067, 0.00986291])
```

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## The structure of the project is as follows:
```
iQuanta
├── data
│   ├── processed
│   │   └── my_file.pkl
│   └── raw
│       └── my_file.pkl
├── iQuanta
│   ├── __init__.py
│   ├── config.py
│   └── funcs.py
├── notebook
│   └── iQuanta.ipynb
├── scripts
│   ├── __init__.py
│   ├── config.py
│   ├── generate_raw_data.py
│   ├── plot_figures.py
│   └── process_data.py
├── tests
│   ├── test_config.py
│   └── test_funcs.py
├── LICENSE
├── README.md
└── setup.py
```

- data: This directory contains two subdirectories: "processed" and "raw". "processed" is intended for storing processed data files, such as pickled dataframes (my_file.pkl), while "raw" is for storing raw data files.
    - processed/my_file.pkl: A processed data file.
    - raw/my_file.pkl: A raw data file.


- iQuanta: This directory holds the core functionality of the iQuanta module.
    - __init__.py: Marks the directory as a Python package.
    - config.py: Contains configuration settings for the iQuanta module.
    - funcs.py: Includes functions and classes defining the main functionalities of the iQuanta module.

    
- notebook: This directory contains Jupyter notebooks for exploratory analysis and demonstrations related to the iQuanta module.
    - iQuanta.ipynb: Jupyter notebook for iQuanta module usage or demonstrations.

    
- scripts: This directory contains Python scripts for various tasks related to the iQuanta module, such as data processing or visualization.
    - __init__.py: Marks the directory as a Python package.
    - config.py: Configuration settings specific to the scripts.
    - generate_raw_data.py: Script for generating raw data.
    - plot_figures.py: Script for plotting figures.
    - process_data.py: Script for processing data.

    
- tests: This directory contains unit tests for the iQuanta module.
    - test_config.py: Unit tests for the configuration module.
    - test_funcs.py: Unit tests for the functions module.

    
- LICENSE: The license file for the project.

- README.md: The README file providing an overview of the project, its purpose, and how to use it.

- setup.py: The setup script for installing the iQuanta module as a Python package.

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## Contributing

Thank you for considering contributing to our project! We welcome contributions from the community to help improve our project and make it even better. To ensure a smooth contribution process, please follow these guidelines:

1. **Fork the Repository**: Fork our repository to your GitHub account and clone it to your local machine.

2. **Branching Strategy**: Create a new branch for your contribution. Use a descriptive branch name that reflects the purpose of your changes.

3. **Code Style**: Follow our coding standards and style guidelines. Make sure your code adheres to the existing conventions to maintain consistency across the project.

4. **Pull Request Process**:
    Before starting work, check the issue tracker to see if your contribution aligns with any existing issues or feature requests.
    Create a new branch for your contribution and make your changes.
    Commit your changes with clear and descriptive messages explaining the purpose of each commit.
    Once you're ready to submit your changes, push your branch to your forked repository.
    Submit a pull request to the main repository's develop branch. Provide a detailed description of your changes and reference any relevant issues or pull requests.

5. **Code Review**: Expect feedback and review from our maintainers or contributors. Address any comments or suggestions provided during the review process.

6. **Testing**: Ensure that your contribution is properly tested. Write unit tests or integration tests as necessary to validate your changes. Make sure all tests pass before submitting your pull request.

7. **Documentation**: Update the project's documentation to reflect your changes. Include any necessary documentation updates, such as code comments, README modifications, or user guides.

8. **License Agreement**: By contributing to our project, you agree to license your contributions under the terms of the project's license (MIT License).

9. **Be Respectful**: Respect the opinions and efforts of other contributors. Maintain a positive and collaborative attitude throughout the contribution process.

We appreciate your contributions and look forward to working with you to improve our project! If you have any questions or need further assistance, please don't hesitate to reach out to us.

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## Credits

- **Author:** [Farhad Razi](https://github.com/fraziphy)

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## License

This project is licensed under the [MIT License](LICENSE)

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## Contact

- **Contact information:** [email](farhad.razi.1988@gmail.com)
