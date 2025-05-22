# Linear Multi-View Acyclic Model

This repository contains the algorithms PRaLiNE and MICaDo (both variants), which estimate a causal order and causal effect matrices from multiple datasets at once.

## How to install

From within your local repository, run

```bash
pip install -e .
```

## Example files

Two example notebooks are given in ```examples```.
The first one, ```example_micado.ipynb``` shows how to use MICaDo-ML and MICaDo-J in the contexts of shared causal ordering or multiple causal orderings. 
The second one, ```example_praline.ipynb``` shows how to use PRaLiNE in the context of shared causal ordering.

## Reproduce simulation studies

The code to reproduce the simulation studies from figures 1, 2, and 4 to 7 is in ```simulation_studies```.

## Reproduce Cam-CAN experiments

The code to reproduce the Cam-CAN experiments from figures 3 and 8 to 10 is in ```real_data_experiments```.