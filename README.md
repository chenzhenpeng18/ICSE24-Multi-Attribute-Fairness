# [ICSE 2024] Fairness Improvement with Multiple Protected Attributes: How Far Are We?

Welcome to visit the homepage of our ICSE'24 paper entitled "Fairness Improvement with Multiple Protected Attributes: How Far Are We?". The homepage contains scripts and data, as well as installation instructions, intermediate results, and a replication guideline.

## Docker image
In order to facilitate the replication of our study, we provide a docker image, which can be downloaded from xx. The docker image includes all the required libraries, datasets, and dependencies for this study. Based on the docker image, you can directly replicate our study following the instructions in the Reproduction and Step-by-step Guide sections.

To use the docker image to replicate our study, you need to 

(1) install docker (https://www.docker.com/).

(2) use docker to load the image using the command: 
	
	docker load -i icse24.tar

(3) excute the image using the commands:

	docker run --name icse24_test -idt icse24

	docker exec -it icse24_test /bin/bash

(4) activate the conda visual environment aif360 using the command:
	
	conda activate aif360

We have included all the code and data of our study in the ```root/ICSE24``` folder of the docker image. You can run the code according to our instructions in the Reproduction and Step-by-step Guide sections.


## Datasets

We have provided the original datasets that we use in this folder. The download link of each dataset is provided as a reference in the paper. We use the data processing scripts provided by [previous work](https://ieeexplore.ieee.org/document/9951398) to process the datasets. We have included the original datasets, the data processing scripts, and the processed datasets in the ```Dataset``` folder.


## Scripts and results

* ```Analysis_code/``` contains the scripts for producing the results for all RQs. You can reproduce all the results based on the intermediate results provided by us by running ```rq1.py```, ```rq2.py```, ```rq3_1.py```, ```rq3_2.py```, and ```rq4.py```.

* ```Fair360/``` and ```Fair3602/``` contain the scripts for implementing fairness improvement methods from the ML community and MAAT proposed by [Chen et al.](https://dl.acm.org/doi/10.1145/3540250.3549093) at ESEC/FSE 2022. The two folders contain the code for dealing a single protected attribute and multiple protected attributes, respectively.

* ```FairMask/``` and ```FairMask2/``` contain the scripts for implementing FairMask, a fairness improvement method proposed by [Peng et al.](https://ieeexplore.ieee.org/document/9951398) at IEEE TSE 2022. The two folders contain the code for dealing a single protected attribute and multiple protected attributes, respectively.

* ```Fair-SMOTE/``` and ```Fair-SMOTE2/``` contains code for implementing Fair-SMOTE, a fairness improvement method proposed by [Chakraborty et al.](https://doi.org/10.1145/3468264.3468537) at ESEC/FSE 2021.

* ```Fairea_multi/``` contains the scripts of the benchmarking tool namely Fairea.

* ```Cal_baseline/``` contains the scripts for generating trade-off baselines using Fairea.

* ```Fairea_baseline_multi/``` contains the generated baseline data.

* ```aif360.zip``` contains the scripts (provided by [Zhang and Sun](https://github.com/zhangmengling/Adaptive_Fairness_Improvement) at ESEC/FSE 2022) of adapting fairness improvement methods from the ML community to make them applicable to multiple protected attributes.
  
* ```Results/``` contains the raw results of applying each fairness improvement method in each scenario with a single protected attribute. Each file in this folder has 21 columns, with the first column indicating the metric, and the next 20 columns the metric values of 20 runs.

* ```Results_multi/``` contains the raw results of applying each fairness improvement method in each scenario with multiple protected attributes. Each file in this folder has 21 columns, with the first column indicating the metric, and the next 20 columns the metric values of 20 runs.

## Reproduction 
You can directly reproduce all the results for all our research questions (RQs) based on the intermediate results provided by us.

### RQ1 (Impact on Unconsidered Protected Attributes)
The results of RQ1 are shown in Table 5 and Table 6. You can reproduce the results as follows:
```
cd 
cd ICSE24/Analysis_code
python rq1.py
```
The commands output `rq1_result`, which provides the results of Table 5 and Table 6.

### RQ2 (Intersectional Fairness Improvement)
The results of RQ2 are shown in Table 7 and Table 8. You can reproduce the results as follows:
```
python rq2.py
```
The commands output `rq2_result`, which provides the results of Table 7 and Table 8.

### RQ3 (Fairness-performance Trade-off)
The results of RQ3 are shown in Table 9 and Figure 1. You can reproduce the results as follows:
```
python rq3_1.py
python rq3_2.py
```
The commands output `rq3_1_result` and `rq3_2_result`, which provide the results of Table 9 and Figure 1, respectively.

### RQ4 (Applicability)
The results of RQ4 are shown in Figure 2. You can reproduce the results as follows:
```
python rq4.py
```
The commands output `rq4_result`, which provides the results of Figure 2(a) and Figure 2(b).

## Step-by-step Guide
You can also reproduce the results from scratch. We provide the step-by-step guide on how to reproduce the intermediate results and obtain the results for RQs based on them.

### RQ1 (Impact on Unconsidered Protected Attributes): How do existing fairness improvement methods affect the fairness regarding unconsidered protected attributes? 
This RQ investigates the negative side effect of single-attribute fairness improvement by studying its impact on fairness regarding the unconsidered protected attributes. 

(1) To answer this RQ, we need to run the code of each fairness improvement method on tasks with single protected attributes. Each fairness improvement method supports three arguments: `-d` configures the dataset; `-c` configures the machine learning algorithm; `-p` configures the protected attribute. We apply each fairness improvement method for 5*2*4=40 (dataset, protected attribute, ML algorithm) tasks. We can achieve this by running the code as follows:

```
cd
cd ICSE24/Fair360
python default.py -d adult -c lr -p sex
python default.py -d adult -c rf -p sex
python default.py -d adult -c svm -p sex
python default_dl.py -d adult -c dl -p sex
python default.py -d adult -c lr -p race
python default.py -d adult -c rf -p race
python default.py -d adult -c svm -p race
python default_dl.py -d adult -c dl -p race
python default.py -d compas -c lr -p sex
python default.py -d compas -c rf -p sex
python default.py -d compas -c svm -p sex
python default_dl.py -d compas -c dl -p sex
python default.py -d compas -c lr -p race
python default.py -d compas -c rf -p race
python default.py -d compas -c svm -p race
python default_dl.py -d compas -c dl -p race
python default.py -d default -c lr -p sex
python default.py -d default -c rf -p sex
python default.py -d default -c svm -p sex
python default_dl.py -d default -c dl -p sex
python default.py -d default -c lr -p age
python default.py -d default -c rf -p age
python default.py -d default -c svm -p age
python default_dl.py -d default -c dl -p age
python default.py -d mep1 -c lr -p sex
python default.py -d mep1 -c rf -p sex
python default.py -d mep1 -c svm -p sex
python default_dl.py -d mep1 -c dl -p sex
python default.py -d mep1 -c lr -p race
python default.py -d mep1 -c rf -p race
python default.py -d mep1 -c svm -p race
python default_dl.py -d mep1 -c dl -p race
python default.py -d mep2 -c lr -p sex
python default.py -d mep2 -c rf -p sex
python default.py -d mep2 -c svm -p sex
python default_dl.py -d mep2 -c dl -p sex
python default.py -d mep2 -c lr -p race
python default.py -d mep2 -c rf -p race
python default.py -d mep2 -c svm -p race
python default_dl.py -d mep2 -c dl -p race
python rw.py -d adult -c lr -p sex
python rw.py -d adult -c rf -p sex
python rw.py -d adult -c svm -p sex
python rw_dl.py -d adult -c dl -p sex
python rw.py -d adult -c lr -p race
python rw.py -d adult -c rf -p race
python rw.py -d adult -c svm -p race
python rw_dl.py -d adult -c dl -p race
python rw.py -d compas -c lr -p sex
python rw.py -d compas -c rf -p sex
python rw.py -d compas -c svm -p sex
python rw_dl.py -d compas -c dl -p sex
python rw.py -d compas -c lr -p race
python rw.py -d compas -c rf -p race
python rw.py -d compas -c svm -p race
python rw_dl.py -d compas -c dl -p race
python rw.py -d default -c lr -p sex
python rw.py -d default -c rf -p sex
python rw.py -d default -c svm -p sex
python rw_dl.py -d default -c dl -p sex
python rw.py -d default -c lr -p age
python rw.py -d default -c rf -p age
python rw.py -d default -c svm -p age
python rw_dl.py -d default -c dl -p age
python rw.py -d mep1 -c lr -p sex
python rw.py -d mep1 -c rf -p sex
python rw.py -d mep1 -c svm -p sex
python rw_dl.py -d mep1 -c dl -p sex
python rw.py -d mep1 -c lr -p race
python rw.py -d mep1 -c rf -p race
python rw.py -d mep1 -c svm -p race
python rw_dl.py -d mep1 -c dl -p race
python rw.py -d mep2 -c lr -p sex
python rw.py -d mep2 -c rf -p sex
python rw.py -d mep2 -c svm -p sex
python rw_dl.py -d mep2 -c dl -p sex
python rw.py -d mep2 -c lr -p race
python rw.py -d mep2 -c rf -p race
python rw.py -d mep2 -c svm -p race
python rw_dl.py -d mep2 -c dl -p race
python dir.py -d adult -c lr -p sex
python dir.py -d adult -c rf -p sex
python dir.py -d adult -c svm -p sex
python dir_dl.py -d adult -c dl -p sex
python dir.py -d adult -c lr -p race
python dir.py -d adult -c rf -p race
python dir.py -d adult -c svm -p race
python dir_dl.py -d adult -c dl -p race
python dir.py -d compas -c lr -p sex
python dir.py -d compas -c rf -p sex
python dir.py -d compas -c svm -p sex
python dir_dl.py -d compas -c dl -p sex
python dir.py -d compas -c lr -p race
python dir.py -d compas -c rf -p race
python dir.py -d compas -c svm -p race
python dir_dl.py -d compas -c dl -p race
python dir.py -d default -c lr -p sex
python dir.py -d default -c rf -p sex
python dir.py -d default -c svm -p sex
python dir_dl.py -d default -c dl -p sex
python dir.py -d default -c lr -p age
python dir.py -d default -c rf -p age
python dir.py -d default -c svm -p age
python dir_dl.py -d default -c dl -p age
python dir.py -d mep1 -c lr -p sex
python dir.py -d mep1 -c rf -p sex
python dir.py -d mep1 -c svm -p sex
python dir_dl.py -d mep1 -c dl -p sex
python dir.py -d mep1 -c lr -p race
python dir.py -d mep1 -c rf -p race
python dir.py -d mep1 -c svm -p race
python dir_dl.py -d mep1 -c dl -p race
python dir.py -d mep2 -c lr -p sex
python dir.py -d mep2 -c rf -p sex
python dir.py -d mep2 -c svm -p sex
python dir_dl.py -d mep2 -c dl -p sex
python dir.py -d mep2 -c lr -p race
python dir.py -d mep2 -c rf -p race
python dir.py -d mep2 -c svm -p race
python dir_dl.py -d mep2 -c dl -p race
python meta.py -d adult -c lr -p sex
python meta.py -d adult -c lr -p race
python meta.py -d compas -c lr -p sex
python meta.py -d compas -c lr -p race
python meta.py -d default -c lr -p sex
python meta.py -d default -c lr -p age
python meta.py -d mep1 -c lr -p sex
python meta.py -d mep1 -c lr -p race
python meta.py -d mep2 -c lr -p sex
python meta.py -d mep2 -c lr -p race
python adv.py -d adult -c lr -p sex
python adv.py -d adult -c lr -p race
python adv.py -d compas -c lr -p sex
python adv.py -d compas -c lr -p race
python adv.py -d default -c lr -p sex
python adv.py -d default -c lr -p age
python adv.py -d mep1 -c lr -p sex
python adv.py -d mep1 -c lr -p race
python adv.py -d mep2 -c lr -p sex
python adv.py -d mep2 -c lr -p race
python pr.py -d adult -c lr -p sex
python pr.py -d adult -c lr -p race
python pr.py -d compas -c lr -p sex
python pr.py -d compas -c lr -p race
python pr.py -d default -c lr -p sex
python pr.py -d default -c lr -p age
python pr.py -d mep1 -c lr -p sex
python pr.py -d mep1 -c lr -p race
python pr.py -d mep2 -c lr -p sex
python pr.py -d mep2 -c lr -p race
python eop.py -d adult -c lr -p sex
python eop.py -d adult -c rf -p sex
python eop.py -d adult -c svm -p sex
python eop_dl.py -d adult -c dl -p sex
python eop.py -d adult -c lr -p race
python eop.py -d adult -c rf -p race
python eop.py -d adult -c svm -p race
python eop_dl.py -d adult -c dl -p race
python eop.py -d compas -c lr -p sex
python eop.py -d compas -c rf -p sex
python eop.py -d compas -c svm -p sex
python eop_dl.py -d compas -c dl -p sex
python eop.py -d compas -c lr -p race
python eop.py -d compas -c rf -p race
python eop.py -d compas -c svm -p race
python eop_dl.py -d compas -c dl -p race
python eop.py -d default -c lr -p sex
python eop.py -d default -c rf -p sex
python eop.py -d default -c svm -p sex
python eop_dl.py -d default -c dl -p sex
python eop.py -d default -c lr -p age
python eop.py -d default -c rf -p age
python eop.py -d default -c svm -p age
python eop_dl.py -d default -c dl -p age
python eop.py -d mep1 -c lr -p sex
python eop.py -d mep1 -c rf -p sex
python eop.py -d mep1 -c svm -p sex
python eop_dl.py -d mep1 -c dl -p sex
python eop.py -d mep1 -c lr -p race
python eop.py -d mep1 -c rf -p race
python eop.py -d mep1 -c svm -p race
python eop_dl.py -d mep1 -c dl -p race
python eop.py -d mep2 -c lr -p sex
python eop.py -d mep2 -c rf -p sex
python eop.py -d mep2 -c svm -p sex
python eop_dl.py -d mep2 -c dl -p sex
python eop.py -d mep2 -c lr -p race
python eop.py -d mep2 -c rf -p race
python eop.py -d mep2 -c svm -p race
python eop_dl.py -d mep2 -c dl -p race
python ceo.py -d adult -c lr -p sex
python ceo.py -d adult -c rf -p sex
python ceo.py -d adult -c svm -p sex
python ceo_dl.py -d adult -c dl -p sex
python ceo.py -d adult -c lr -p race
python ceo.py -d adult -c rf -p race
python ceo.py -d adult -c svm -p race
python ceo_dl.py -d adult -c dl -p race
python ceo.py -d compas -c lr -p sex
python ceo.py -d compas -c rf -p sex
python ceo.py -d compas -c svm -p sex
python ceo_dl.py -d compas -c dl -p sex
python ceo.py -d compas -c lr -p race
python ceo.py -d compas -c rf -p race
python ceo.py -d compas -c svm -p race
python ceo_dl.py -d compas -c dl -p race
python ceo.py -d default -c lr -p sex
python ceo.py -d default -c rf -p sex
python ceo.py -d default -c svm -p sex
python ceo_dl.py -d default -c dl -p sex
python ceo.py -d default -c lr -p age
python ceo.py -d default -c rf -p age
python ceo.py -d default -c svm -p age
python ceo_dl.py -d default -c dl -p age
python ceo.py -d mep1 -c lr -p sex
python ceo.py -d mep1 -c rf -p sex
python ceo.py -d mep1 -c svm -p sex
python ceo_dl.py -d mep1 -c dl -p sex
python ceo.py -d mep1 -c lr -p race
python ceo.py -d mep1 -c rf -p race
python ceo.py -d mep1 -c svm -p race
python ceo_dl.py -d mep1 -c dl -p race
python ceo.py -d mep2 -c lr -p sex
python ceo.py -d mep2 -c rf -p sex
python ceo.py -d mep2 -c svm -p sex
python ceo_dl.py -d mep2 -c dl -p sex
python ceo.py -d mep2 -c lr -p race
python ceo.py -d mep2 -c rf -p race
python ceo.py -d mep2 -c svm -p race
python ceo_dl.py -d mep2 -c dl -p race
python roc.py -d adult -c lr -p sex
python roc.py -d adult -c rf -p sex
python roc.py -d adult -c svm -p sex
python roc_dl.py -d adult -c dl -p sex
python roc.py -d adult -c lr -p race
python roc.py -d adult -c rf -p race
python roc.py -d adult -c svm -p race
python roc_dl.py -d adult -c dl -p race
python roc.py -d compas -c lr -p sex
python roc.py -d compas -c rf -p sex
python roc.py -d compas -c svm -p sex
python roc_dl.py -d compas -c dl -p sex
python roc.py -d compas -c lr -p race
python roc.py -d compas -c rf -p race
python roc.py -d compas -c svm -p race
python roc_dl.py -d compas -c dl -p race
python roc.py -d default -c lr -p sex
python roc.py -d default -c rf -p sex
python roc.py -d default -c svm -p sex
python roc_dl.py -d default -c dl -p sex
python roc.py -d default -c lr -p age
python roc.py -d default -c rf -p age
python roc.py -d default -c svm -p age
python roc_dl.py -d default -c dl -p age
python roc.py -d mep1 -c lr -p sex
python roc.py -d mep1 -c rf -p sex
python roc.py -d mep1 -c svm -p sex
python roc_dl.py -d mep1 -c dl -p sex
python roc.py -d mep1 -c lr -p race
python roc.py -d mep1 -c rf -p race
python roc.py -d mep1 -c svm -p race
python roc_dl.py -d mep1 -c dl -p race
python roc.py -d mep2 -c lr -p sex
python roc.py -d mep2 -c rf -p sex
python roc.py -d mep2 -c svm -p sex
python roc_dl.py -d mep2 -c dl -p sex
python roc.py -d mep2 -c lr -p race
python roc.py -d mep2 -c rf -p race
python roc.py -d mep2 -c svm -p race
python roc_dl.py -d mep2 -c dl -p race
python maat.py -d adult -c lr -p sex
python maat.py -d adult -c rf -p sex
python maat.py -d adult -c svm -p sex
python maat_dl.py -d adult -c dl -p sex
python maat.py -d adult -c lr -p race
python maat.py -d adult -c rf -p race
python maat.py -d adult -c svm -p race
python maat_dl.py -d adult -c dl -p race
python maat.py -d compas -c lr -p sex
python maat.py -d compas -c rf -p sex
python maat.py -d compas -c svm -p sex
python maat_dl.py -d compas -c dl -p sex
python maat.py -d compas -c lr -p race
python maat.py -d compas -c rf -p race
python maat.py -d compas -c svm -p race
python maat_dl.py -d compas -c dl -p race
python maat.py -d default -c lr -p sex
python maat.py -d default -c rf -p sex
python maat.py -d default -c svm -p sex
python maat_dl.py -d default -c dl -p sex
python maat.py -d default -c lr -p age
python maat.py -d default -c rf -p age
python maat.py -d default -c svm -p age
python maat_dl.py -d default -c dl -p age
python maat.py -d mep1 -c lr -p sex
python maat.py -d mep1 -c rf -p sex
python maat.py -d mep1 -c svm -p sex
python maat_dl.py -d mep1 -c dl -p sex
python maat.py -d mep1 -c lr -p race
python maat.py -d mep1 -c rf -p race
python maat.py -d mep1 -c svm -p race
python maat_dl.py -d mep1 -c dl -p race
python maat.py -d mep2 -c lr -p sex
python maat.py -d mep2 -c rf -p sex
python maat.py -d mep2 -c svm -p sex
python maat_dl.py -d mep2 -c dl -p sex
python maat.py -d mep2 -c lr -p race
python maat.py -d mep2 -c rf -p race
python maat.py -d mep2 -c svm -p race
python maat_dl.py -d mep2 -c dl -p race

cd
cd ICSE24/FairMask
python Fairmask.py -d adult -c lr -p sex
python Fairmask.py -d adult -c rf -p sex
python Fairmask.py -d adult -c svm -p sex
python Fairmask_dl.py -d adult -c dl -p sex
python Fairmask.py -d adult -c lr -p race
python Fairmask.py -d adult -c rf -p race
python Fairmask.py -d adult -c svm -p race
python Fairmask_dl.py -d adult -c dl -p race
python Fairmask.py -d compas -c lr -p sex
python Fairmask.py -d compas -c rf -p sex
python Fairmask.py -d compas -c svm -p sex
python Fairmask_dl.py -d compas -c dl -p sex
python Fairmask.py -d compas -c lr -p race
python Fairmask.py -d compas -c rf -p race
python Fairmask.py -d compas -c svm -p race
python Fairmask_dl.py -d compas -c dl -p race
python Fairmask.py -d default -c lr -p sex
python Fairmask.py -d default -c rf -p sex
python Fairmask.py -d default -c svm -p sex
python Fairmask_dl.py -d default -c dl -p sex
python Fairmask.py -d default -c lr -p age
python Fairmask.py -d default -c rf -p age
python Fairmask.py -d default -c svm -p age
python Fairmask_dl.py -d default -c dl -p age
python Fairmask.py -d mep1 -c lr -p sex
python Fairmask.py -d mep1 -c rf -p sex
python Fairmask.py -d mep1 -c svm -p sex
python Fairmask_dl.py -d mep1 -c dl -p sex
python Fairmask.py -d mep1 -c lr -p race
python Fairmask.py -d mep1 -c rf -p race
python Fairmask.py -d mep1 -c svm -p race
python Fairmask_dl.py -d mep1 -c dl -p race
python Fairmask.py -d mep2 -c lr -p sex
python Fairmask.py -d mep2 -c rf -p sex
python Fairmask.py -d mep2 -c svm -p sex
python Fairmask_dl.py -d mep2 -c dl -p sex
python Fairmask.py -d mep2 -c lr -p race
python Fairmask.py -d mep2 -c rf -p race
python Fairmask.py -d mep2 -c svm -p race
python Fairmask_dl.py -d mep2 -c dl -p race

cd
cd ICSE24/FairSMOTE
python FairSMOTE.py -d adult -c lr -p sex
python FairSMOTE.py -d adult -c rf -p sex
python FairSMOTE.py -d adult -c svm -p sex
python FairSMOTE_dl.py -d adult -c dl -p sex
python FairSMOTE.py -d adult -c lr -p race
python FairSMOTE.py -d adult -c rf -p race
python FairSMOTE.py -d adult -c svm -p race
python FairSMOTE_dl.py -d adult -c dl -p race
python FairSMOTE.py -d compas -c lr -p sex
python FairSMOTE.py -d compas -c rf -p sex
python FairSMOTE.py -d compas -c svm -p sex
python FairSMOTE_dl.py -d compas -c dl -p sex
python FairSMOTE.py -d compas -c lr -p race
python FairSMOTE.py -d compas -c rf -p race
python FairSMOTE.py -d compas -c svm -p race
python FairSMOTE_dl.py -d compas -c dl -p race
python FairSMOTE.py -d default -c lr -p sex
python FairSMOTE.py -d default -c rf -p sex
python FairSMOTE.py -d default -c svm -p sex
python FairSMOTE_dl.py -d default -c dl -p sex
python FairSMOTE.py -d default -c lr -p age
python FairSMOTE.py -d default -c rf -p age
python FairSMOTE.py -d default -c svm -p age
python FairSMOTE_dl.py -d default -c dl -p age
python FairSMOTE.py -d mep1 -c lr -p sex
python FairSMOTE.py -d mep1 -c rf -p sex
python FairSMOTE.py -d mep1 -c svm -p sex
python FairSMOTE_dl.py -d mep1 -c dl -p sex
python FairSMOTE.py -d mep1 -c lr -p race
python FairSMOTE.py -d mep1 -c rf -p race
python FairSMOTE.py -d mep1 -c svm -p race
python FairSMOTE_dl.py -d mep1 -c dl -p race
python FairSMOTE.py -d mep2 -c lr -p sex
python FairSMOTE.py -d mep2 -c rf -p sex
python FairSMOTE.py -d mep2 -c svm -p sex
python FairSMOTE_dl.py -d mep2 -c dl -p sex
python FairSMOTE.py -d mep2 -c lr -p race
python FairSMOTE.py -d mep2 -c rf -p race
python FairSMOTE.py -d mep2 -c svm -p race
python FairSMOTE_dl.py -d mep2 -c dl -p race

```

As a result, we can obtain the results of each fairness improvement for 40 (dataset, protected attribute, ML algorithm) combinations. The raw result for each combination is then moved to the `Results/` folder. For example, in this folder, `maat_lr_adult_sex.txt` contains the results of MAAT for the (adult, sex, lr) combination. Each file in the folder has 21 columns, with the first column indicating the metric and the next 20 columns the metric values of 20 runs.


(2) We can obtain the results of RQ1 (i.e., Table 5 and Table 6) as follows:

```
cd 
cd ICSE24/Analysis_code
python rq1.py
```

### RQ2 (Intersectional Fairness Improvement): What intersectional fairness do existing fairness improvement methods achieve when considering multiple protected attributes?
This RQ evaluates the effectiveness of state-of-the-art fairness improvement methods in improving intersectional fairness.

(1) We first apply each fairness improvement method to tasks with multiple protected attributes. Each fairness improvement method supports three arguments: `-d` configures the dataset; `-c` configures the machine learning algorithm. We apply each fairness improvement method for 5*4=20 (dataset, ML algorithm) tasks. We can achieve this by running the code as follows:

```
cd 
cd ICSE24/Fair3602
python default.py -d adult -c lr
python default.py -d adult -c rf
python default.py -d adult -c svm
python default_dl.py -d adult -c dl
python default.py -d compas -c lr
python default.py -d compas -c rf
python default.py -d compas -c svm
python default_dl.py -d compas -c dl
python default.py -d default -c lr
python default.py -d default -c rf
python default.py -d default -c svm
python default_dl.py -d default -c dl
python default.py -d mep1 -c lr
python default.py -d mep1 -c rf
python default.py -d mep1 -c svm
python default_dl.py -d mep1 -c dl
python default.py -d mep2 -c lr
python default.py -d mep2 -c rf
python default.py -d mep2 -c svm
python default_dl.py -d mep2 -c dl
python rw.py -d adult -c lr
python rw.py -d adult -c rf
python rw.py -d adult -c svm
python rw_dl.py -d adult -c dl
python rw.py -d compas -c lr
python rw.py -d compas -c rf
python rw.py -d compas -c svm
python rw_dl.py -d compas -c dl
python rw.py -d default -c lr
python rw.py -d default -c rf
python rw.py -d default -c svm
python rw_dl.py -d default -c dl
python rw.py -d mep1 -c lr
python rw.py -d mep1 -c rf
python rw.py -d mep1 -c svm
python rw_dl.py -d mep1 -c dl
python rw.py -d mep2 -c lr
python rw.py -d mep2 -c rf
python rw.py -d mep2 -c svm
python rw_dl.py -d mep2 -c dl
python dir.py -d adult -c lr
python dir.py -d adult -c rf
python dir.py -d adult -c svm
python dir_dl.py -d adult -c dl
python dir.py -d compas -c lr
python dir.py -d compas -c rf
python dir.py -d compas -c svm
python dir_dl.py -d compas -c dl
python dir.py -d default -c lr
python dir.py -d default -c rf
python dir.py -d default -c svm
python dir_dl.py -d default -c dl
python dir.py -d mep1 -c lr
python dir.py -d mep1 -c rf
python dir.py -d mep1 -c svm
python dir_dl.py -d mep1 -c dl
python dir.py -d mep2 -c lr
python dir.py -d mep2 -c rf
python dir.py -d mep2 -c svm
python dir_dl.py -d mep2 -c dl
python meta.py -d adult -c lr
python meta.py -d compas -c lr
python meta.py -d default -c lr
python meta.py -d mep1 -c lr
python meta.py -d mep2 -c lr
python adv.py -d adult -c lr
python adv.py -d compas -c lr
python adv.py -d default -c lr
python adv.py -d mep1 -c lr
python adv.py -d mep2 -c lr
python pr.py -d adult -c lr
python pr.py -d compas -c lr
python pr.py -d default -c lr
python pr.py -d mep1 -c lr
python pr.py -d mep2 -c lr
python eop.py -d adult -c lr
python eop.py -d adult -c rf
python eop.py -d adult -c svm
python eop_dl.py -d adult -c dl
python eop.py -d compas -c lr
python eop.py -d compas -c rf
python eop.py -d compas -c svm
python eop_dl.py -d compas -c dl
python eop.py -d default -c lr
python eop.py -d default -c rf
python eop.py -d default -c svm
python eop_dl.py -d default -c dl
python eop.py -d mep1 -c lr
python eop.py -d mep1 -c rf
python eop.py -d mep1 -c svm
python eop_dl.py -d mep1 -c dl
python eop.py -d mep2 -c lr
python eop.py -d mep2 -c rf
python eop.py -d mep2 -c svm
python eop_dl.py -d mep2 -c dl
python ceo.py -d adult -c lr
python ceo.py -d adult -c rf
python ceo.py -d adult -c svm
python ceo_dl.py -d adult -c dl
python ceo.py -d compas -c lr
python ceo.py -d compas -c rf
python ceo.py -d compas -c svm
python ceo_dl.py -d compas -c dl
python ceo.py -d default -c lr
python ceo.py -d default -c rf
python ceo.py -d default -c svm
python ceo_dl.py -d default -c dl
python ceo.py -d mep1 -c lr
python ceo.py -d mep1 -c rf
python ceo.py -d mep1 -c svm
python ceo_dl.py -d mep1 -c dl
python ceo.py -d mep2 -c lr
python ceo.py -d mep2 -c rf
python ceo.py -d mep2 -c svm
python ceo_dl.py -d mep2 -c dl
python roc.py -d adult -c lr
python roc.py -d adult -c rf
python roc.py -d adult -c svm
python roc_dl.py -d adult -c dl
python roc.py -d compas -c lr
python roc.py -d compas -c rf
python roc.py -d compas -c svm
python roc_dl.py -d compas -c dl
python roc.py -d default -c lr
python roc.py -d default -c rf
python roc.py -d default -c svm
python roc_dl.py -d default -c dl
python roc.py -d mep1 -c lr
python roc.py -d mep1 -c rf
python roc.py -d mep1 -c svm
python roc_dl.py -d mep1 -c dl
python roc.py -d mep2 -c lr
python roc.py -d mep2 -c rf
python roc.py -d mep2 -c svm
python roc_dl.py -d mep2 -c dl
python maat.py -d adult -c lr
python maat.py -d adult -c rf
python maat.py -d adult -c svm
python maat_dl.py -d adult -c dl
python maat.py -d compas -c lr
python maat.py -d compas -c rf
python maat.py -d compas -c svm
python maat_dl.py -d compas -c dl
python maat.py -d default -c lr
python maat.py -d default -c rf
python maat.py -d default -c svm
python maat_dl.py -d default -c dl
python maat.py -d mep1 -c lr
python maat.py -d mep1 -c rf
python maat.py -d mep1 -c svm
python maat_dl.py -d mep1 -c dl
python maat.py -d mep2 -c lr
python maat.py -d mep2 -c rf
python maat.py -d mep2 -c svm
python maat_dl.py -d mep2 -c dl

cd 
cd ICSE24/FairMask2
python Fairmask.py -d adult -c lr
python Fairmask.py -d adult -c rf
python Fairmask.py -d adult -c svm
python Fairmask_dl.py -d adult -c dl
python Fairmask.py -d compas -c lr
python Fairmask.py -d compas -c rf
python Fairmask.py -d compas -c svm
python Fairmask_dl.py -d compas -c dl
python Fairmask.py -d default -c lr
python Fairmask.py -d default -c rf
python Fairmask.py -d default -c svm
python Fairmask_dl.py -d default -c dl
python Fairmask.py -d mep1 -c lr
python Fairmask.py -d mep1 -c rf
python Fairmask.py -d mep1 -c svm
python Fairmask_dl.py -d mep1 -c dl
python Fairmask.py -d mep2 -c lr
python Fairmask.py -d mep2 -c rf
python Fairmask.py -d mep2 -c svm
python Fairmask_dl.py -d mep2 -c dl

cd 
cd ICSE24/FairSMOTE2
python FairSMOTE.py -d adult -c lr
python FairSMOTE.py -d adult -c rf
python FairSMOTE.py -d adult -c svm
python FairSMOTE_dl.py -d adult -c dl
python FairSMOTE.py -d compas -c lr
python FairSMOTE.py -d compas -c rf
python FairSMOTE.py -d compas -c svm
python FairSMOTE_dl.py -d compas -c dl
python FairSMOTE.py -d default -c lr
python FairSMOTE.py -d default -c rf
python FairSMOTE.py -d default -c svm
python FairSMOTE_dl.py -d default -c dl
python FairSMOTE.py -d mep1 -c lr
python FairSMOTE.py -d mep1 -c rf
python FairSMOTE.py -d mep1 -c svm
python FairSMOTE_dl.py -d mep1 -c dl
python FairSMOTE.py -d mep2 -c lr
python FairSMOTE.py -d mep2 -c rf
python FairSMOTE.py -d mep2 -c svm
python FairSMOTE_dl.py -d mep2 -c dl

```

As a result, we can obtain the results of each fairness improvement for 20 (dataset, ML algorithm) combinations. The raw result for each combination is then moved to the `Results_multi/` folder. For example, in this folder, `maat_lr_adult_multi.txt` contains the results of MAAT for the (adult, lr) combination. Each file in the folder has 21 columns, with the first column indicating the metric and the next 20 columns the metric values of 20 runs.


(2) We can obtain the results of RQ2 (i.e., Table 7 and Table 8) as follows:

```
cd 
cd ICSE24/Analysis_code
python rq2.py
```


### RQ3 (Fairness-performance Trade-off): What fairness-performance trade-off do existing fairness improvement methods achieve when considering multiple protected attributes?

(1) We first answer RQ3.1: Does the application of existing methods to improve fairness for multiple protected attributes lead to significantly greater performance reduction compared to improving fairness for a single attribute?

Since in RQ1 and RQ2, we can calculate the effects of each fairness improvement method on machine learning performance when dealing with single or multiple protected attributes. The results are included in `Results/` and `Results_multi/`. We can directly obtain the results of RQ3.1 (i.e., Table 9) as follows:

```
cd 
cd ICSE24/Analysis_code
python rq3_1.py
```

(2) We next answer RQ3.2: Which trade-off effectiveness levels do existing fairness improvement methods fall into according to Fairea?

For each (dataset, ML algorithm) combination, we use Fairea to construct the fairness-performance trade-off baseline.
```
cd
cd ICSE24/Cal_baseline
python cal_baselinepoints_multi.py -d adult -c lr
python cal_baselinepoints_multi.py -d adult -c rf
python cal_baselinepoints_multi.py -d adult -c svm
python cal_baselinepoints_multi_dl.py -d adult -c dl
python cal_baselinepoints_multi.py -d compas -c lr
python cal_baselinepoints_multi.py -d compas -c rf
python cal_baselinepoints_multi.py -d compas -c svm
python cal_baselinepoints_multi_dl.py -d compas -c dl
python cal_baselinepoints_multi.py -d default -c lr
python cal_baselinepoints_multi.py -d default -c rf
python cal_baselinepoints_multi.py -d default -c svm
python cal_baselinepoints_multi_dl.py -d default -c dl
python cal_baselinepoints_multi.py -d mep1 -c lr
python cal_baselinepoints_multi.py -d mep1 -c rf
python cal_baselinepoints_multi.py -d mep1 -c svm
python cal_baselinepoints_multi_dl.py -d mep1 -c dl
python cal_baselinepoints_multi.py -d mep2 -c lr
python cal_baselinepoints_multi.py -d mep2 -c rf
python cal_baselinepoints_multi.py -d mep2 -c svm
python cal_baselinepoints_multi_dl.py -d mep2 -c dl
```

The baselines for each (dataset, ML algorithm) combination is included in the `Fairea_baseline_multi/` folder. For example, `adult_lr_baseline.txt` contains the baseline for the (adult, lr) combination. Each file in the folder has 12 columns, with the first column indicating the ML performance or fairness metric, the second column the metric values of the original model, the next 10 columns the metric values of 10 pseudo models (with different mutation degrees).

Then we can obtain the results of RQ3.2 (i.e., Figure 1) as follows:

```
cd 
cd ICSE24/Analysis_code
python rq3_2.py
```

### RQ4 (Applicability): How well do existing fairness improvement methods apply to different decision tasks, ML models, and fairness and performance metrics, when dealing with multiple protected attributes?

Then we can obtain the results of RQ4 (i.e., Figure 2) as follows:

```
cd 
cd ICSE24/Analysis_code
python rq4.py
```

## Declaration
Thanks to the authors of existing fairness improvement methods for open source, to facilitate our implementation of this paper. Therefore, when using our code or data for your work, please also consider citing their papers, including [AIF360](https://arxiv.org/abs/1810.01943), [Fair-SMOTE](https://doi.org/10.1145/3468264.3468537), [MAAT](https://dl.acm.org/doi/10.1145/3540250.3549093), [FairMask](https://ieeexplore.ieee.org/document/9951398), and [Fairea](https://doi.org/10.1145/3468264.3468565).

## Citation
Please consider citing the following paper when using our code or data.
```
@inproceedings{zhenpengicse24,
  title={Fairness Improvement with Multiple Protected Attributes: How Far Are We?},
  author={Zhenpeng Chen and Jie M. Zhang and Federica Sarro and Mark Harman},
  booktitle={Proceedings of the 46th International Conference on Software Engineering, ICSE'24},
  year={2024}
}
```
