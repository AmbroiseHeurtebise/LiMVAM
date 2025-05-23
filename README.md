# Linear Multi-View Acyclic Model (LiMVAM)

This repository contains the PRaLiNE and MICaDo algorithms (including the MICaDo-ML and MICaDo-J variants), which estimate causal order and causal effect matrices from multiple datasets simultaneously.

## How to install

From within your local repository, run

```bash
pip install -e .
```

## Main files

The code for the two algorithms can be found in ```limvam/praline.py``` and ```limvam/micado.py```.

## Example files

Two example notebooks are provided in the ```examples``` folder.
The first one, ```example_micado.ipynb``` shows how to use MICaDo-ML and MICaDo-J in contexts of shared or multiple causal orderings. 
The second one, ```example_praline.ipynb``` shows how to use PRaLiNE in the shared causal ordering context.

## Reproduce simulation studies

The code to reproduce the simulation studies is located in the ```simulation_studies``` folder. 

Specifically:
- The file to run figure 1 is in ```simulation_studies/runs/run_timepoints_in_xaxis.py```
- The file to run figure 2 is in ```simulation_studies/runs/run_execution_time.py```
- The file to run figure 4 is in ```simulation_studies/runs/run_p_in_xaxis.py```
- The file to run figure 5 is in ```simulation_studies/runs/run_noise_in_xaxis.py```
- The file to run figure 6 is in ```simulation_studies/runs/run_noise_diversity.py```
- The file to run figure 7 is in ```simulation_studies/runs/run_sparsity_of_Ti.py```

The results of the simulations are stored in ```simulation_studies/results```.

The plotting files are in ```simulation_studies/plotting```, and the obtained figures are in ```simulation_studies/figures```.

## Reproduce Cam-CAN experiments

The code to reproduce the Cam-CAN experiments shown in figures 3, 8, 9, and 10 is located in the ```real_data_experiments``` folder.
However, running these experiments requires downloading the Cam-CAN dataset and adapting some paths.

The folder ```real_data_experiments/1_preprocessing``` is used to preprocess the Cam-CAN data, and stores the obtained envelopes of oscillatory signals in ```real_data_experiments/2_data_envelopes```. These envelopes are then used in ```real_data_experiments/3_runs```, and the results of the runs are stored in ```real_data_experiments/4_results```. Finally, these results are used in ```real_data_experiments/5_plotting``` to produce the desired figures.