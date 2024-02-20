# setup.py

# The setup.py is adapted from https://stackoverflow.com/questions/4740473/setup-py-examples
from distutils.core import setup
from setuptools import find_packages
import os


# User-friendly description from README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    long_description = ''

setup(
    # Name of the package
    name='iQuanta',

    # Packages to include into the distribution
    packages=find_packages('.'), 

    # Start with a small number and increase it with every change you make
    # https://semver.org
    version='1.0.0',

    # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    # For example: MIT
    license='MIT License',

    # Short description of your library
    description='A module to quantify information content in neural signals.',

    # Long description of your library
    long_description = long_description,
    long_description_context_type = 'text/markdown',

    # Your name
    author='Farhad Razi', 

    # Your email
    author_email='farhad.razi.1988@gmail.com',     

    # Either the link to your github or to your website
    url='https://github.com/fraziphy/',

    # Link from which the project can be downloaded
    download_url='https://github.com/fraziphy/iQuanta',

    # List of keyword arguments
    keywords=["information quantification","neural signals"],

    # List of packages to install with this one
    install_requires=[],

    # https://pypi.org/classifiers/
    classifiers=[]  
)


#from setuptools import setup

#with open("README.md", "r") as fh:
    #long_description = fh.read()

#setup(
   #name='iQuanta',
   #version='1.0',
   #description='A module to quantify information content in neural signals.',
   #license="MIT License",
   #author='Farhad-Razi',
   #long_description=long_description,
   #author_email='farhad.razi.1988@gmail.com'
#)
