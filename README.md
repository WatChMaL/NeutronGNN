# Water Cherenkov Machine Learning for neutron identification with graph neural nets


## Description

Python implementation of the training engine and framework to build, train and test graph neutral net models for Water Cherenkov Detectors.

## Table of Contents

### 1. [Directory Layout](#directory_layout)
### 2. [Installation](#installation)

## Directory Layout <a id="directory_layout"></a>

```bash
.
+-- plot_utils                          # Tools for visualizing model performance and dataset features
  +-- mpmt_visual.py
  +-- notebook_utils.py
  +-- plot_utils.py
+-- root_utils                          # Tools for interacting with the ROOT files from the WCSim simulations
  +-- pos_utils.py
+-- README.md                           # README documentation for the repository
```

## Installation <a id="installation"></a>

### Requirements

The following Python standard, machine learning and deep learning libraries are required for the functionality of the framework :

1. [PyTorch](https://pytorch.org/)
2. [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
3. [NumPy](https://www.numpy.org/)
4. [Matplotlib](https://matplotlib.org/users/installing.html)

To download the repository use :

`git clone https://github.com/WatChMaL/NeutronGNN.git`
