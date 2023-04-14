import pymc3 as pm
import numpy as np
# import random

def sample(model, samples, chains, tune):
    """
        Parameters
        -----------
        model:    PyMC3 model object
        samples:  Integer - number of samples to draw from posterior predictive dist
        chains:   Integer - number of MCMC chains 
        tune:     Integer - number of MCMC samples ....<TODO>
        Returns            
        --------
        PyMC3 trace object
    """    
    with model:
        trace = pm.sample(samples, chains=chains, tune=tune)
    return trace

def sample_posterior_predictive(model, trace, samples, var_names = None):#destand_dict,
    """
        Parameters
        -----------
        model:             PyMC3 model object
        trace:             PyMC3 trace
        samples:           Integer number of samples to draw from posterior predictive dist
        var_names:         List of var_names to be sampled from
        Returns            
        --------
        Dict (<observed_var>: Numpy Array of pp samples of same shape as corresponding likelihood)
    """
    if var_names is None:
        deterministic = [var.name for var in model.deterministics]
        # var_names = [var.name for var in model.observed_RVs]  
        # var_names.extend([var.name[:-1] for var in model.observed_RVs if var.name[:-1] in deterministic])
        var_names = [var.name[:-1] for var in model.observed_RVs if var.name[:-1] in deterministic]   
    with model:
         pp_samples = pm.sample_posterior_predictive(trace,
                                                     samples = samples, 
                                                     var_names = var_names)
    return pp_samples

def simulate_atomic_intervention(intervention, model, trace, samples, var_names = None):#, destand_dict
    """
        Parameters
        -----------
        intervention:      Dict (<var>: List of values to set <var> equal to)
        model:             PyMC3 model object
        trace:             PyMC3 trace
        samples:           Integer number of samples to draw from posterior predictive dist
        ##destand_dict: Dict (<observed_var>: Tuple (<observed_var> std,<observed_var> mean))
        Returns            
        --------
        Dict (<observed_var>: List of pp samples)
        Dict(<observed_var>: numpy array of shape (i_value,chain,draw,obs_id))
    """    
    samples_i = {}
    for var_i,values in intervention.items():
        with model:
            pm.set_data({'x_'+var_i: 0.})
        if len(values) > 2:
            c = abs(values[1] - values[0])            
        for i in values:
            with model:
                pm.set_data({'y_'+var_i: i})
                pm.set_data({'c_'+var_i: c})
                print("i_var",var_i,"c",c, "i_value",i)
            pp_samples = sample_posterior_predictive(model, 
                                                     trace, 
                                                     samples, 
                                                     var_names)  #,  destand_dict  
            # ## add samples to returned dict
            # for var in pp_samples:  
            #     if var not in samples_i:
            #         samples_i[var] = []
            #     samples_i[var].append(pp_samples[var].flatten())
            ## add samples to returned dict
            for var in pp_samples:                  
                if var not in samples_i:                
                    samples_i[var] = np.expand_dims(pp_samples[var], axis=(0,1))## add 1st dim for i_value and 2nd for chain
                else:
                    pp_samples[var] = np.expand_dims(pp_samples[var], axis=(0,1))## add 1st dim for i_value and 2nd for chain
                    samples_i[var] = np.concatenate((samples_i[var], pp_samples[var]), axis=0)
        with model:
            pm.set_data({'x_'+var_i: 1.})
            pm.set_data({'y_'+var_i: 0.})
            pm.set_data({'c_'+var_i: 1.})
    return samples_i

def simulate_shift_intervention(intervention, model, trace, samples, var_names = None):#, destand_dict
    """
        Parameters
        -----------
        intervention:      Dict (<var>: List of values to set <var> equal to)
        model:             PyMC3 model object
        trace:             PyMC3 trace
        samples:           Integer number of samples to draw from posterior predictive dist
        destand_dict: Dict (<observed_var>: Tuple (<observed_var> std,<observed_var> mean))
        Returns            
        --------
        Dict (<observed_var>: List of pp samples)
    """    
    samples_i = {}    
    for var_i,values in intervention.items():
        for i in values:
            with model:
                pm.set_data({'shift_'+var_i: i})
            pp_samples = sample_posterior_predictive(model, 
                                                     trace, 
                                                     samples, 
                                                     var_names)  #,  destand_dict  
            # ## add samples to returned dict
            # for var in pp_samples:  
            #     if var not in samples_i:
            #         samples_i[var] = []
            #     samples_i[var].append(pp_samples[var].flatten())
            ## add samples to returned dict
            for var in pp_samples:                  
                if var not in samples_i:                
                    samples_i[var] = np.expand_dims(pp_samples[var], axis=(0,1))
                else:
                    pp_samples[var] = np.expand_dims(pp_samples[var], axis=(0,1))
                    samples_i[var] = np.concatenate((samples_i[var], pp_samples[var]), axis=0)
        with model:
            pm.set_data({'shift_'+var_i: 0.})
    return samples_i

def simulate_variance_intervention(intervention, model, trace, samples, var_names = None):#, destand_dict
    """
        Parameters
        -----------
        intervention:      Dict (<var>: List of values to set <var> equal to)
        model:             PyMC3 model object
        trace:             PyMC3 trace
        samples:           Integer number of samples to draw from posterior predictive dist
        destand_dict: Dict (<observed_var>: Tuple (<observed_var> std,<observed_var> mean))
        Returns            
        --------
        Dict (<observed_var>: List of pp samples)
    """    
    samples_i = {}    
    for var_i,values in intervention.items():
        for i in values:
            with model:
                pm.set_data({'variance_'+var_i: i})
            pp_samples = sample_posterior_predictive(model, 
                                                     trace, 
                                                     samples, 
                                                     var_names)  #,  destand_dict  
            # ## add samples to returned dict
            # for var in pp_samples:  
            #     if var not in samples_i:
            #         samples_i[var] = []
            #     samples_i[var].append(pp_samples[var].flatten())
            ## add samples to returned dict
            for var in pp_samples:                  
                if var not in samples_i:                
                    samples_i[var] = np.expand_dims(pp_samples[var], axis=(0,1))
                else:
                    pp_samples[var] = np.expand_dims(pp_samples[var], axis=(0,1))
                    samples_i[var] = np.concatenate((samples_i[var], pp_samples[var]), axis=0)
        with model:
            pm.set_data({'variance_'+var_i: 1.})
    return samples_i
