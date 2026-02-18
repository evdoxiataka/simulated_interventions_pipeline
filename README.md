# Probabilistic Simulator of Interventions

### Pipeline to Simulate Interventions on Causal Structures Probabilistically

Probabilistic modelling, Bayesian inference, and PyMC are used.

Causal models are expressed in PyMC probabilistic statements: a likelihood is defined for each of the observed variables in the causal model and prior distributions are defined for the parameters of the likelihoods.

 ![PYMC](https://github.com/user-attachments/assets/b4fb7578-5996-46fe-9e33-91826735ab45)


Then, we run Bayesian inference: a PyMC MCMC (Markov Chain Monte Carlo) sampler is used to sample from the posterior distribution of the model and generate posterior predictive samples for the observed variables of the model.

The definition of likelihoods include appropriate mathematical transformations that allow 
1. the activation of causal interventions (atomic, shift or variance interventions according to Witty et al.) and consequently,
2. the post-intervention required structural adaptations of the model's definition
on-the-fly and once the inference has been run, and without the manual intervention of the user and the requirement.

Here's a diagram of the tool's inner workings:

![DiagramSim](https://github.com/user-attachments/assets/fbcf43d2-7f74-47b6-8f66-c84ceb0e2a8f)

### Contents of Repository
- **simulate_interventions_insomnia.ipynb** : demonstrates how to use pipeline for the insomnia-anxiety-tiredness problem
- **utils/model.py** : contains the code for automatic transformation of causal model to PyMC3 code defining a probabilistic linear regression model
- **utils/simulator.py** : contains the code for simulating interventions using PyMC3's posterior predictive sampler
