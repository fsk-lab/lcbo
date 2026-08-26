This repository contains all scripts to reproduce the experiments described in the paper "MoSeS Enables Autonomous Chromatography Operation through Programmable Analytical Expertise".

## Repo Structure

* `lcbo`: Routines for running HPLC-MS experiments, and for experiment planning with Bayesian Optimization. 
* `scripts`: Scripts and experiment configurations to run the experiments described in the paper.


## Dependencies

All code is written in Python (tested and stable for `python >= 3.9`). 

Instrument control, data analysis, and HPLC simulation use the `moses` library, which is available through `pip` (`pip install moses-hplc`) or from the Aspuru-Guzik group's [*GitLab* repository](https://gitlab.com/aspuru-guzik-group/self-driving-lab/instruments/moses). Detailed documentation for `moses` can be found in the [package documentation](https://moses-hplc.readthedocs.io).

For Bayesian optimization and advanced data interpretation, the following additional libraries are required:
* `botorch` (developed with `botorch == 0.10.0`)
* `laplace-bayesopt` (developed with `laplace-bayesopt == 0.1.7`)
* `botier` (developed with `botier == 1.0.0`)
* `rdkit` (developed with `rdkit == 2025.3.6`)

## Data and Results

All data and results from the paper are provided on Zenodo at [https://doi.org/10.5281/zenodo.22109729](https://doi.org/10.5281/zenodo.22109729).
