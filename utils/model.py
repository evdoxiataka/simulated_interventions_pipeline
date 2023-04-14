import pymc3 as pm
import theano.tensor as T
import numpy as np
from utils.utils.constants import N, uniform_magn

## EVALUATION MODEL
def create_model(dag, observations):
    """
        Parameters
        ----------
        dag           Dict <var_name: List of vars> 
                      We also insert first in the dict the independent variables 
                      with an empty List as a value, then the parents of 
                      the independent variables and so on. The order really matters
                      
        observations  Dict <varname: Numpy array of observeations>
    """
    ## instantiate a model object
    coords = {'obs_id':[i for i in range(len(list(observations.values())[0]))]}
    model = pm.Model(coords=coords)
    ##
    var_obs = {}## variable names for observations in upper case
    var_inf = {}## variable names for inference in lower case
    ##
    for var in dag:
        ## add data variables
        var_obs[var] = var.upper()
        if var.isupper():
            var_inf[var] = var.lower()
        else:
            var_inf[var] = var
        with model: 
            ## OBSERVATIONS
            pm.Data(var_obs[var], observations[var])
            pm.Data(var_obs[var]+'_mean', observations[var].mean())## mean value of observations for var
            pm.Data(var_obs[var]+'_std', observations[var].std())## std of observations for var
            pm.Data(var_obs[var]+'_st', (observations[var]-model.named_vars[var_obs[var]+'_mean'].get_value()) / model.named_vars[var_obs[var]+'_std'].get_value())## standardized observations for var
            ## VARIABLES TO CONTROL 
            ## MODEL CHANGES AFTER INTERVENTIONS
            pm.Data('x_'+var_inf[var],1.)## switch on/off variable of likelihood of var
            pm.Data('y_'+var_inf[var],0.)## variable to set var equal to atomic intervention value
            pm.Data('shift_'+var_inf[var],0.)## variable to shift mean of var
            pm.Data('variance_'+var_inf[var],1.)## variable to scale the variance of var
            ## VARIABLES TO ADD NOISE TO INTERVENTION VARIABLE
            pm.Data('c_'+var_inf[var],1.)## always positive number
            # pm.Data('noise_low_'+var_inf[var], model.named_vars['y_'+var_inf[var]].get_value() - model.named_vars['c_'+var_inf[var]].get_value())
            # pm.Data('noise_high_'+var_inf[var], model.named_vars['y_'+var_inf[var]].get_value() + model.named_vars['c_'+var_inf[var]].get_value())
        ## IF VAR HAS PARENTS   
        if len(dag[var]):
            with model:
                ## PRIORS
                pm.Normal('a_'+var_inf[var], mu=0, sd=1)
                pm.HalfNormal('sigma_'+var_inf[var], sd=1) 
                # pm.Uniform('noised_y_'+var_inf[var], model.named_vars['noise_low_'+var_inf[var]].get_value(), model.named_vars['noise_high_'+var_inf[var]].get_value())
                ## REGRESSION
                reg = model.named_vars['a_'+var_inf[var]]
                for p in dag[var]:
                    pm.Normal('b_'+var_inf[var]+'_'+var_inf[p], mu=0, sd=1)
                    reg = reg + model.named_vars['b_'+var_inf[var]+'_'+var_inf[p]]*model.named_vars[var_inf[p]]            
                ## LIKELIHOOD
                pm.Normal(var_inf[var]+'_',
                          mu = reg + model.named_vars['shift_'+var_inf[var]],
                          sd = model.named_vars['sigma_'+var_inf[var]]*model.named_vars['variance_'+var_inf[var]],
                          observed = model.named_vars[var_obs[var]+'_st'],
                          dims = 'obs_id')
                ## PRIOR for NOISE
                pm.Uniform('noised_y_'+var_inf[var],
                           model.named_vars['y_'+var_inf[var]] - model.named_vars['c_'+var_inf[var]],
                           model.named_vars['y_'+var_inf[var]] + model.named_vars['c_'+var_inf[var]], shape = N)## shape same as number of observations
                ## x_var * (var_std*var + var_mean) + y_var_noised
                pm.Deterministic(var_inf[var],
                                 model.named_vars['x_'+var_inf[var]]*(model.named_vars[var_obs[var]+'_std']*model.named_vars[var_inf[var]+'_'] + model.named_vars[var_obs[var]+'_mean']) + pm.math.switch(model.named_vars['y_'+var_inf[var]], model.named_vars['noised_y_'+var_inf[var]], model.named_vars['y_'+var_inf[var]]),
                                 dims = 'obs_id')
        ## IF VAR HAS NO PARENTS
        else:
            with model:
                ## PRIORS
                pm.Normal('mu_'+var_inf[var],
                          mu=0,
                          sd=1)
                pm.HalfNormal('sigma_'+var_inf[var],
                              sd=1)
                # pm.Uniform('noised_y_'+var_inf[var],
                #            model.named_vars['y_'+var_inf[var]] - model.named_vars['c_'+var_inf[var]],
                #            model.named_vars['y_'+var_inf[var]] + model.named_vars['c_'+var_inf[var]])
                # pm.Uniform('noised_y_'+var_inf[var], model.named_vars['noise_low_'+var_inf[var]].get_value(), model.named_vars['noise_high_'+var_inf[var]].get_value())
                ## LIKELIHOOD
                pm.Normal(var_inf[var]+'_',
                          mu = model.named_vars['mu_'+var_inf[var]] + model.named_vars['shift_'+var_inf[var]],
                          sd = model.named_vars['sigma_'+var_inf[var]]*model.named_vars['variance_'+var_inf[var]],
                          observed = model.named_vars[var_obs[var]+'_st'],
                          dims = 'obs_id')
                ##
                pm.Uniform('noised_y_'+var_inf[var],
                           (model.named_vars['y_'+var_inf[var]] - model.named_vars['c_'+var_inf[var]]),
                           (model.named_vars['y_'+var_inf[var]] + model.named_vars['c_'+var_inf[var]]), shape = N)
                ## x_var * (var_std*var + var_mean) + y_var_noised
                pm.Deterministic(var_inf[var], 
                                 model.named_vars['x_'+var_inf[var]]*(model.named_vars[var_obs[var]+'_std']*model.named_vars[var_inf[var]+'_'] + model.named_vars[var_obs[var]+'_mean']) + pm.math.switch(model.named_vars['y_'+var_inf[var]], model.named_vars['noised_y_'+var_inf[var]], model.named_vars['y_'+var_inf[var]]),
                                 dims = 'obs_id')
    return model