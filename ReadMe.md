VeRecycle <td align="center" valign="top" width="33.33%"><img src="https://github.com/SUMI-lab/VeRecycle/blob/main/VeRecycle-logo.png?raw=true" width="100px;"/></td>
=============================

This repository contains the supplementary code for the paper:

> Sterre Lutz, Matthijs T.J. Spaan, Anna Lukina. "VeRecycle: Reclaiming Guarantees from Probabilistic Certificates for Stochastic Dynamical Systems after Change."
IJCAI (2025).

To prepare input certificates for VeRecycle and compare VeRecycle's performance to a baseline of re-certification, 
some files are reused or adapted (clearly indicated at the top of those files) from 
[this repository](https://github.com/LAVA-LAB/neural_stochastic_control), which
contains supplementary code for the paper:
> Badings, Thom, et al. "Learning-Based Verification of Stochastic Dynamical Systems with Neural Network Policies." 
arXiv preprint arXiv:2406.00826 (2024).

## How to Install
Experiments for the paper were run using Python 3.12. Install the required packages as stated below. While GPU
acceleration is not necessary to run VeRecycle, it is **strongly recommended** when reproducing the input or running the
baselines.
### Requirements
#### With GPU acceleration
```
conda create -n verecycle python=3.12
conda activate verecycle
pip install -r requirements_gpu.txt
```
#### Without GPU acceleration
```
conda create -n verecycle python=3.12
conda activate verecycle
pip install -r requirements_cpu.txt
```

### LaTeX
The code is set up to process plot text as LaTeX commands by default. This requires a LaTeX installation along with 
correctly configured environment variables. If you prefer to disable this feature, you can use the `--no-latex` 
flag.

## Reproduce Experiments

### Step 1: Prepare Initial Policies and Certificates
For our experiments, we used the policies and certificates stored in the folder `logger/ninerooms_original`. 
These certified controllers can be reproduced using the command below for `SEED` values 1-5.
```
python synthesize_compositional_controller.py --seed [SEED] --logger_prefix ninerooms_original --model NineRooms --silent --no-latex 
``` 
### Step 2: Run VeRecycle and Baselines
To reproduce the experiments from the paper, run the following command for `SEED` values 1-5 and 
`METHOD` values: 
- `infimum` (VeRecycle), 
- `informed_check` (Baseline B1), 
- `binary_search` (Baseline B2), and 
- `binary_search_from_scratch` (Baseline B3).
```
python run.py --method [METHOD] --seed [SEED] --load_folder_compositional [FOLDER-IN] --experiment_folder [FOLDER-OUT] --model NineRooms --silent --no-latex
```
To run the experiments on our prepared inputs, use 
`logger/ninerooms_original/model=NineRooms_date=2024-11-23_14-19-45__seed=[SEED]` as `FOLDER-IN`.

### Step 3: Reproduce Paper Figures and Tables
Running `plot_paper_figures.py` plots the three figures (1(a), 1(b), and 2) from the paper and saves them to `figures/`.

Running `parse_experiment_results.py` parses the experiment results into Table 1 and 2 from the paper, printing them 
in the console. 
