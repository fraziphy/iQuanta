# iQuanta
This code package is developed to formulate a novel framework to quantify the information content in neural data.


## The structure of the project is as follows:
```
project
├── README.md
├── project
│   ├── __init__.py
│   ├── config.py
│   ├── custom_funcs.py
│   └── custom_funcs.py
├── notebooks
├── scripts
├── data
├── setup.py
└── tests
    └── __init__.py
```


To validate that the scripts and the required libraries are installed on the local machine you can get started by running tests on the pythons scripts within the project and scripts directories. To do so, install the python libray nose2 and run it from the command line.

```
$ pip install nose2
$ python -m nose2
```


To install the package as a python module in your Jupyter notebook, please run the following command:

```
!pip install git+ssh://git@github.com/fraziphy/iQuanta.git
```

To impoirt the function to quantofy the information content in the neural signals, proceed as follows:


```
from iQuanta.custom_funcs import Information_Content as iquanta
```
