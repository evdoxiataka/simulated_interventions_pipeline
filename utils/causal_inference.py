import numpy as np

from utils.simulator import sample, sample_posterior_predictive, simulate_atomic_intervention, simulate_shift_intervention, simulate_variance_intervention
from utils.utils.constants import x_range_mag, hdi_magn, slider_tick_num, samples, chains, tune, pp_samples, sim_i_pp_samples, shift_i_magn


def get_intervention_values_cont(observations):
    """
        Parameters
        ----------
            observations: Numpy array of observations for a var
        Returns
        -------
            interventions: Numpy array of intervention values for a var
    """
#         hdi = az.hdi(observations[var], hdi_prob=.95)
    range = []
    range.append(observations.min())
    range.append(observations.max())
    range_init = range[1]-range[0]
    range[0] = range[0] - hdi_magn * abs(range_init)
    range[1] = range[1] + hdi_magn * abs(range_init)
    return np.linspace(range[0], range[1], slider_tick_num).tolist()

def get_shift_intervention_values(observations):
    """
        Parameters
        ----------
            observations: Numpy array of observations for a var
        Returns
        -------
            interventions: Numpy array of intervention values for a var
    """
    range_init = observations.max()-observations.min()
    range = []
    range.append(-(shift_i_magn * abs(range_init)))
    range.append(shift_i_magn * abs(range_init))
    return np.linspace(range[0], range[1], slider_tick_num).tolist()

def get_intervention_values_disc(observations):
    """
        Parameters
        ----------
            observations: Numpy array of observations for a var
        Returns
        -------
            interventions: Numpy array of intervention values for a var
    """
    return np.unique(observations).tolist()

def get_causal_inference(observations, causal_dags, models):
    ## Get intervention values per observed variable
    atomic_interventions = {}
    shift_interventions = {}
    variance_interventions = {}
    ## Fit model to each DAG, estimate inference
    causal_inference = {}
    causal_inference['dags'] = {}
    causal_inference["vars"] = {}
    for i,dag in enumerate(causal_dags):
        causal_inference['dags'][i] = {}
        causal_inference['dags'][i]['dag'] = dag
        ## MODEL
        model = models[i]
        causal_inference['dags'][i]['model'] = model
        ## INFERENCE
        trace = sample(model, samples, chains, tune)
        causal_inference['dags'][i]['trace'] = trace
        ## PP SAMPLES        
        causal_inference['dags'][i]['pp_samples'] = sample_posterior_predictive(model, trace, pp_samples)
        ## INTERVENTIONS       
        causal_inference['dags'][i]['ia_samples'] = {}## Dict <i_var>: Dict <var>: List of numpy array of samples after intervention
        causal_inference['dags'][i]['is_samples'] = {}
        causal_inference['dags'][i]['iv_samples'] = {}
        for i_var in dag:
            if i_var not in atomic_interventions:
                causal_inference["vars"][i_var] = {}
                cls = [c.__name__ for c in model.named_vars[i_var+"_"].distribution.__class__.__mro__]
                if "Continuous" in cls:
                    atomic_interventions[i_var] = get_intervention_values_cont(observations[i_var])
                    causal_inference["vars"][i_var]["type"] = "Continuous"
                else:
                    atomic_interventions[i_var] = get_intervention_values_disc(observations[i_var])
                    causal_inference["vars"][i_var]["type"] = "Discrete" 
                if "Normal" in cls:
                    causal_inference["vars"][i_var]["dist"] = "Normal"
#                     shift_interventions[i_var] = np.linspace(1., 20., 20).tolist() 
                    shift_interventions[i_var] = get_shift_intervention_values(observations[i_var])
                    variance_interventions[i_var] = np.linspace(0.2, 0.4, slider_tick_num).tolist() 
                else:
                    causal_inference["vars"][i_var]["dist"] = ""
            ## Atomic
            causal_inference['dags'][i]['ia_samples'][i_var] = simulate_atomic_intervention({i_var:atomic_interventions[i_var]}, model, trace, sim_i_pp_samples) 
            if causal_inference["vars"][i_var]["dist"] == "Normal":
                ## Shift
                causal_inference['dags'][i]['is_samples'][i_var] = simulate_shift_intervention({i_var:shift_interventions[i_var]}, model, trace, sim_i_pp_samples) 
                ## Variance
                causal_inference['dags'][i]['iv_samples'][i_var] = simulate_variance_intervention({i_var:variance_interventions[i_var]}, model, trace, sim_i_pp_samples)            
    ## estimate var x_range across models
    estimate_x_range_across_models(observations, causal_inference)
    ## 
    return atomic_interventions, shift_interventions, variance_interventions, causal_inference

def estimate_x_range_across_models(observations, causal_inference):
    causal_inference['x_range'] = {}
    causal_inference['x_range']['pp_samples'] = {}
    causal_inference['x_range']['ia_samples'] = {}
    causal_inference['x_range']['is_samples'] = {}
    causal_inference['x_range']['iv_samples'] = {}
    for var in observations:
        x_range_min = []
        x_range_max = []
        x_range_ia_var_min = {}
        x_range_ia_var_max = {}
        x_range_is_var_min = {}
        x_range_is_var_max = {}
        x_range_iv_var_min = {}
        x_range_iv_var_max = {}
        for i in causal_inference['dags']:
            ## pp_samples
            if var in causal_inference['dags'][i]['pp_samples']:
                x_range_min.append(causal_inference['dags'][i]['pp_samples'][var].min())
                x_range_max.append(causal_inference['dags'][i]['pp_samples'][var].max())
            ## atomic intervention
            for i_var in causal_inference['dags'][i]['ia_samples']:
                if var in causal_inference['dags'][i]['ia_samples'][i_var]:
                    if i_var not in x_range_ia_var_min:
                        x_range_ia_var_min[i_var] = []
                        x_range_ia_var_max[i_var] = []
                    x_range_ia_var_min[i_var].append(np.array(causal_inference['dags'][i]['ia_samples'][i_var][var]).min())
                    x_range_ia_var_max[i_var].append(np.array(causal_inference['dags'][i]['ia_samples'][i_var][var]).max())
            ## shift
            for i_var in causal_inference['dags'][i]['is_samples']:
                if var in causal_inference['dags'][i]['is_samples'][i_var]:
                    if i_var not in x_range_is_var_min:
                        x_range_is_var_min[i_var] = []
                        x_range_is_var_max[i_var] = []
                        x_range_iv_var_min[i_var] = []
                        x_range_iv_var_max[i_var] = []
                    x_range_is_var_min[i_var].append(np.array(causal_inference['dags'][i]['is_samples'][i_var][var]).min())
                    x_range_is_var_max[i_var].append(np.array(causal_inference['dags'][i]['is_samples'][i_var][var]).max())
                    x_range_iv_var_min[i_var].append(np.array(causal_inference['dags'][i]['iv_samples'][i_var][var]).min())
                    x_range_iv_var_max[i_var].append(np.array(causal_inference['dags'][i]['iv_samples'][i_var][var]).max())
        ## var x_range of pp_samples
        if len(x_range_min):
            var_min = min(x_range_min)
            var_max = max(x_range_max)
            causal_inference['x_range']['pp_samples'][var] = (var_min-x_range_mag*abs(var_max-var_min),var_max+x_range_mag*abs(var_max-var_min))
        ## var x_range of ia_samples
        for i_var in x_range_ia_var_min:
            var_min = min(x_range_ia_var_min[i_var])
            var_max = max(x_range_ia_var_max[i_var])
            if i_var not in causal_inference['x_range']['ia_samples']:
                causal_inference['x_range']['ia_samples'][i_var] = {}
            causal_inference['x_range']['ia_samples'][i_var][var] = (var_min-x_range_mag*abs(var_max-var_min),var_max+x_range_mag*abs(var_max-var_min))
        ## var x_range of is_samples
        for i_var in x_range_is_var_min:
            ## shift
            var_min = min(x_range_is_var_min[i_var])
            var_max = max(x_range_is_var_max[i_var])
            if i_var not in causal_inference['x_range']['is_samples']:
                causal_inference['x_range']['is_samples'][i_var] = {}
            causal_inference['x_range']['is_samples'][i_var][var] = (var_min-x_range_mag*abs(var_max-var_min),var_max+x_range_mag*abs(var_max-var_min))
            ## variance
            var_min = min(x_range_iv_var_min[i_var])
            var_max = max(x_range_iv_var_max[i_var])
            if i_var not in causal_inference['x_range']['iv_samples']:
                causal_inference['x_range']['iv_samples'][i_var] = {}
            causal_inference['x_range']['iv_samples'][i_var][var] = (var_min-x_range_mag*abs(var_max-var_min),var_max+x_range_mag*abs(var_max-var_min))