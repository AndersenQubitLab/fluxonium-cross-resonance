# Code for  *Exploration of Fluxonium Parameters for Capacitive Cross-Resonance Gates*

Simulation data are saved in the `data/` directory by the Jupyter notebooks in `python/notebooks/Calculations/`.
Plottings notebooks in `python/notebooks/Plotting/` use this data to generate figures used for the paper.
The notebooks depend on the package `fluxoniumcr`, which contains more modular simulation and plotting code.
To install it, create a new Python environment and change into the directory `python/fluxoniumcr/` and then run
```
pip install --editable .
```
to install the package as editable.
The package must be installed as editable for the notebooks to be able to find the `data/` directory.
