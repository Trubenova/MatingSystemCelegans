# Investigating Mating System consequences for resistance evolution
This is code for investigating the consequences of mating systems for resistance evolution. 

## WormFunctionsMating.py 
This is the main file containing all the classes (e.g., Experiment, Population) needed to run the simulations. All details of the experiments are encoded here. 
Individual outputs of the simulations are saved in individual files. 

## MatingWormsCluster.ipynb and MatingWormsCluster2Loci.ipynb

These are the two files that can run simulations for situations with 6 and 2 loci, respectively. They define the parameters of the experiment and call the necessary functions to run it. 

## DataMerger.py 
This code merges individual output data repeats together for future manipulation. 

## AnalyzeDataCluster.ipynb

This is the code that takes outputs of DataMerger.py and calculates and plots relevant statistics and observations. 
