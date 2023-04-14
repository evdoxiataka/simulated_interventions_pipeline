# Probabilistic Simulator of Interventions

### Pipeline to Simulate Interventions on Causal Structures Probabilistically

Probabilistic Models, Bayesian Inference, and PyMC3 are used.

- **simulate_interventions_insomnia.ipynb** : demonstrates how to use pipeline for the insomnia-anxiety-tiredness problem
- **utils/model.py** : contains the code for automatic transformation of causal model to PyMC3 code defining a probabilistic linear regression model
- **utils/simulator.py** : contains the code for simulating interventions using PyMC3's posterior predictive sampler
