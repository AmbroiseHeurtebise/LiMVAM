![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

# Linear Multi-View Acyclic Model (LiMVAM)

This repository implements the algorithms introduced in [this paper](https://arxiv.org/pdf/2502.20115).

It provides three main methods: **DirectLiMVAM**, **PairwiseLiMVAM**, and **ICA-LiMVAM** (with its two variants **ICA-LiMVAM-J** and **ICA-LiMVAM-ML**). 

Given multi-view data (multiple datasets describing the same variables across different views), these algorithms aim to:
- Recover a **shared causal ordering** of variables across views
- Estimate the **strength of causal relationships**

All implementations are written in Python and are available in ```limvam```.

## Installation

From within your local repository, run

```bash
pip install -e .
```

## Quick example

```python
from limvam.direct_limvam import direct_limvam
import numpy as np

X = np.random.randn(n_views, n_variables, n_samples)
B, T, P = direct_limvam(X)
```

In the outputs: 
- ```B``` are square matrices contains the strength of causal relationships
- ```T``` are strictly lower triangular matrices
- ```P``` is a permutation matrix that contains the causal ordering

## Getting started

Three example notebooks are provided in the ```examples``` folder. 
They demonstrate how to use each algorithm in practice on simple synthetic data.

## fMRI experiment

This experiment uses data from a [preprocessed dataset](https://github.com/cabal-cmu/Feedback-Discovery).

The dataset contains fMRI recordings from 9 participants who performed a rhyming judgment task. 
Each participant's recordings contain 9 variables: one task regressor and 8 brain regions. 
We apply one of our methods to recover a causal graph between brain regions.

The experiment runs quickly and is available in the ```experiments_fmri``` folder.

## MEG experiments

The code to reproduce the MEG experiments is located in the ```experiments_meg``` folder.
However, running these experiments requires downloading the [Cam-CAN dataset](https://www.sciencedirect.com/science/article/pii/S1053811915008150) and adapting some paths.

## Synthetic experiments

The ```experiments_synthetic``` folder contains several simulation studies evaluating the methods under different scenarios, including: varying the numbers of views, variables, samples, or the noise level.

## Cite

If you use this code in your project, please cite [this paper](https://arxiv.org/pdf/2502.20115):

```bash
Ambroise Heurtebise, Omar Chehab, Pierre Ablin, Alexandre Gramfort, Aapo Hyvärinen
Multi-View Causal Discovery without Non-Gaussianity: Identifiability and Algorithms
arXiv preprint, 2025
```