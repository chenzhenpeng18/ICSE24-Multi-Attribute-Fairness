# Fairness Improvement with Multiple Protected Attributes: How Far Are We?

Welcome to visit the homepage of our paper entitled "Fairness Improvement with Multiple Protected Attributes: How Far Are We?". The homepage contains scripts and data used in this paper.

## Experimental environment

We use Python 3.7 for our experiments. We use the IBM AI Fairness 360 (AIF360) toolkit for implementing bias mitigation methods and computing fairness metrics. 

Installation instructions for Python 3.7 and AIF360 can be found on https://github.com/Trusted-AI/AIF360. That page provides several ways for the installation. We recommend creating a virtual environment for it (as shown below), because AIF360 requires specific versions of many Python packages which may conflict with other projects on your system. If you would like to try other installation ways or encounter any errors during the installation proces, please refer to the page (https://github.com/Trusted-AI/AIF360) for help.

#### Conda

Conda is recommended for all configurations. [Miniconda](https://conda.io/miniconda.html)
is sufficient (see [the difference between Anaconda and
Miniconda](https://conda.io/docs/user-guide/install/download.html#anaconda-or-miniconda)
if you are curious) if you do not already have conda installed.

Then, to create a new Python 3.7 environment, run:

```bash
conda create --name aif360 python=3.7
conda activate aif360
```

The shell should now look like `(aif360) $`. To deactivate the environment, run:

```bash
(aif360)$ conda deactivate
```

The prompt will return to `$ `.

Note: Older versions of conda may use `source activate aif360` and `source
deactivate` (`activate aif360` and `deactivate` on Windows).

### Install with `pip`

To install the latest stable version from PyPI, run:

```bash
pip install 'aif360'
```

[comment]: <> (This toolkit can be installed as follows:)

[comment]: <> (```)

[comment]: <> (pip install aif360)

[comment]: <> (```)

[comment]: <> (More information on installing AIF360 can be found on https://github.com/Trusted-AI/AIF360.)

In addition, we require the following Python packages. Note that TensorFlow is only required for implementing the existing fairness improvement method named ADV. If you do not want to implement this method, you can skip the installation of TensorFlow (the last line of the following commands).
```
pip install sklearn
pip install numpy
pip install shapely
pip install matplotlib
pip install "tensorflow >= 1.13.1, < 2"
pip install --upgrade protobuf==3.20.0
pip install fairlearn
```

## Dataset

You can refer to https://github.com/Trusted-AI/AIF360/tree/master/aif360/data for the raw data files. We use the data processing scripts provided by [previous work](https://ieeexplore.ieee.org/document/9951398) to process the datasets. We have included the datasets and data processing scripts in the ```Dataset``` folder.


## Scripts and results

* ```Analysis_code/``` contains the scripts for producing the results for all RQs. You can reproduce all the results based on the intermediate results provided by us by running ```rq.py```, ```rq2.py```, ```rq3_1.py```, ```rq3_2.py```, and ```rq4.py```.

* ```Fair360/``` and ```Fair3602/``` contain the scripts for implementing the fairness improvement methods from the ML community and MAAT proposed by [Chen et al.](https://dl.acm.org/doi/10.1145/3540250.3549093). They contain the code for dealing a single protected attribute and multiple protected attributes, respectively.

* ```FairMask/``` and ```FairMask2/``` contain the scripts for implementing FairMask, a fairness improvement method proposed by [Peng et al.](https://ieeexplore.ieee.org/document/9951398) in IEEE TSE 2022. They contain the code for dealing a single protected attribute and multiple protected attributes, respectively.

* ```Fair-SMOTE/``` and ```Fair-SMOTE2/``` contains code for implementing Fair-SMOTE, a fairness improvement method proposed by [Chakraborty et al.](https://doi.org/10.1145/3468264.3468537) in ESEC/FSE 2021.

* ```Fairea_multi/``` contains the scripts of the benchmarking tool namely Fairea.

* ```Cal_baseline/``` contains the scripts for generating trade-off baselines using Fairea.

* ```Fairea_baseline_multi/``` contains the generated baseline data.

* ```aif360.zip``` contains the scripts (provided by [Zhang and Sun](https://github.com/zhangmengling/Adaptive_Fairness_Improvement) at ESEC/FSE 2022) of adapting fairness improvement methods from the ML community to make them applicable to multiple protected attributes.
  
* ```Results.zip``` contains the raw results of applying each fairness improvement method in each scenario with a single protected attribute. Each file in this folder has 21 columns, with the first column indicating the metric, and the next 20 columns the metric values of 20 runs.

* ```Results_multi.zip``` contains the raw results of applying each fairness improvement method in each scenario with multiple protected attributes. Each file in this folder has 21 columns, with the first column indicating the metric, and the next 20 columns the metric values of 20 runs.


