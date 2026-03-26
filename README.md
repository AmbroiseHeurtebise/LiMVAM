# Linear Multi-View Acyclic Model (LiMVAM)

This repository contains the algorithms introduced in [this paper](https://arxiv.org/pdf/2502.20115).

There are three algorithms: DirectLiMVAM, PairwiseLiMVAM, and ICA-LiMVAM (with its two variants ICA-LiMVAM-J and ICA-LiMVAM-ML). 
Given a dataset made of multiple views, variables, and samples, these algorithms seek for a causal ordering between variables that is shared across views. They also estimate the strength of the causal effects. 

The three algorithms are written in Python and can be found in ```limvam```.

## How to install

From within your local repository, run

```bash
pip install -e .
```

## Getting started with examples

Three example notebooks are provided in the ```examples``` folder. They show how to use the three algorithms in practice on simple synthetic data.

## fMRI experiment

Data for this experiment come from [this preprocessed dataset](https://github.com/cabal-cmu/Feedback-Discovery).
They consist in fMRI recordings from 9 participants who performed a rhyming judgment task. Each participant's recordings contain 9 variables: one task regressor (Input I) and 8 brain regions (LOCC, ROCC, LIPL, RIPL, LACC, RACC, LIFG, RIFG). We then apply one of our algorithms to recover a causal graph between brain regions.

The experiment runs quickly and can be found in ```experiments_fmri```.

## MEG experiments

The code to reproduce the MEG experiments is located in the ```experiments_meg``` folder.
However, running these experiments requires downloading the [Cam-CAN dataset](https://www.sciencedirect.com/science/article/pii/S1053811915008150) and adapting some paths.

## Synthetic experiments

There are several simulation studies in the ```experiments_synthetic``` folder. They evaluate the methods in various scenarios, such as when varying the numbers of views, variables, samples, or the noise level.

## Cite

If you use this code in your project, please cite [this paper](https://arxiv.org/pdf/2502.20115):

```bash
Ambroise Heurtebise, Omar Chehab, Pierre Ablin, Alexandre Gramfort, Aapo Hyvärinen
Multi-View Causal Discovery without Non-Gaussianity: Identifiability and Algorithms
arXiv preprint, 2025
```