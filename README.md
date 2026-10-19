# IB Physics EE: Coupled Magnetic Oscillator

Python code and data-processing workflow for an IB Physics Extended Essay investigating how pivot separation affects the normal-mode frequencies of a five-pendulum magnetic Newton's cradle.

## Scripts

- `Appendix_B_Tracker_Processing.py` — converts Tracker coordinate data into angular displacement from equilibrium and checks tracked pendulum length.
- `Appendix_C_Fourier_Analysis.py` — performs Fourier analysis, calculates sampling quantities, and identifies candidate spectral peaks using a 3:1 signal-to-noise criterion.
- `Appendix_D_Linearized_Model.py` — calculates analytical normal-mode frequencies and eigenvectors from the linearized coupled-oscillator model.
- `Appendix_E_Nonlinear_Simulation.py` — numerically integrates the nonlinear equations of motion using finite-cylinder magnetic interactions, gravitational torque, viscous damping, and Coulomb friction.
- `Statistics_Model_Comparison.py` — calculates repeated-trial statistics, normalized residuals, percentage discrepancies, and RMSE for model comparison.

## Requirements

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

## Data files

The scripts expect experimental input files such as Tracker coordinate exports, equilibrium coordinates, fitted damping parameters, experimental modal frequencies, and numerical frequency results. File names and expected column headings are defined at the top of each script.
