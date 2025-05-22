# Linear Multi-View Acyclic Model (LiMVAM)

This repository contains the algorithms PRaLiNE and MICaDo (both variants MICaDo-ML and MICaDo-J), which estimate a causal order and causal effect matrices from multiple datasets at once.

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

The code to reproduce the simulation studies is in ```simulation_studies```. 

Specifically:
- the file to run figure 1 is in ```simulation_studies/runs/run_timepoints_in_xaxis.py```
- the file to run figure 2 is in ```simulation_studies/runs/run_execution_time.py```
- the file to run figure 4 is in ```simulation_studies/runs/run_p_in_xaxis.py```
- the file to run figure 5 is in ```simulation_studies/runs/run_noise_in_xaxis.py```
- the file to run figure 6 is in ```simulation_studies/runs/run_noise_diversity.py```
- the file to run figure 7 is in ```simulation_studies/runs/run_sparsity_of_Ti.py```

Then, the results of the simulations are stored in ```simulation_studies/results```.

The plotting files are in ```simulation_studies/plotting```, and the obtained figures are in ```simulation_studies/figures```.

## Reproduce Cam-CAN experiments

The code to reproduce the Cam-CAN experiments from figures 3 and 8 to 10 is in ```real_data_experiments```.
However, running these experiments requires downloading the Cam-CAN dataset, and may require to adapt some paths.

The repository ```real_data_experiments/1_preprocessing``` is used to preprocess Cam-CAN data, and then stores the envelopes of oscillatory signals in ```real_data_experiments/2_data_envelopes```. These envelopes are used in the two files in ```real_data_experiments/3_runs```, and the corresponding results are stored in ```real_data_experiments/4_results```. Finally, these results are used in ```real_data_experiments/5_plotting``` to obtain the desired figures.