import xarray as xr
from arviz_json import get_dag, arviz_to_json
import arviz as az
import numpy as np
import json

from utils.simulator import sample, sample_posterior_predictive, simulate_atomic_intervention, simulate_shift_intervention, simulate_variance_intervention
from utils.utils.constants import slider_tick_num, samples, chains, tune, pp_samples, sim_i_pp_samples
from utils.causal_inference import get_intervention_values_cont, get_shift_intervention_values, get_intervention_values_disc, estimate_x_range_across_models

def get_observed_vars(dag, with_ = True):
    vars = []
    for var in dag:
        if dag[var]['type'] == 'observed' and dag[var]['name'][-1] == "_":
            if with_:
                vars.append(dag[var]['name'])
            else:
                vars.append(dag[var]['name'][:-1])
    return vars

def get_npz_file(model, fileName, simulateInterventions = True):
    ## 
    dag = get_dag(model)
    i_vars_ = get_observed_vars(dag)
    i_vars = get_observed_vars(dag, with_ = False)## names of observed variables on causal dag 
    ##
    trace = sample(model, samples, chains, tune)
    ## PP SAMPLES        
    posterior_predictive = sample_posterior_predictive(model, trace, pp_samples, var_names = i_vars)
    ## CONVERT INTO ARVIZ INFERENCEDATA OBJ
    data = az.from_pymc3(trace=trace, posterior_predictive=posterior_predictive)
    ## Add graph dag
    data.sample_stats.attrs["graph"] = str(dag)
    ## Get intervention values per observed variable
    if simulateInterventions:
        interventions = {}
        ia_samples = {}
        is_samples = {}
        iv_samples = {}
        for i_var_ in i_vars_:
            i_var = i_var_[:-1]
            ## Atomic Intervention
            if dag[i_var_]['distribution']['type'] == "Continuous":
                interventions["ai_"+i_var] = get_intervention_values_cont(data.constant_data[i_var.upper()].values)
            elif dag[i_var_]['distribution']['type'] == "Discrete":
                interventions["ai_"+i_var] = get_intervention_values_disc(data.constant_data[i_var.upper()].values)        
            ia_s = simulate_atomic_intervention({i_var:interventions["ai_"+i_var]}, model, trace, sim_i_pp_samples, var_names = i_vars) 
            for v in ia_s:
                if i_var not in ia_samples:
                    ia_samples[i_var] = np.expand_dims(ia_s[v], axis=0)
                else:
                    ia_s[v] = np.expand_dims(ia_s[v], axis=0)
                    ia_samples[i_var] = np.concatenate((ia_samples[i_var], ia_s[v]), axis=0)

            ## Shift - Variance Intervention
            if dag[i_var_]['distribution']['dist'] == "Normal": 
                interventions["si_"+i_var] = get_shift_intervention_values(data.constant_data[i_var.upper()].values)
                if i_var == "insomnia":
                    interventions["vi_"+i_var] = np.linspace(1.5, 8., slider_tick_num).tolist() 
                elif i_var == "anxiety":
                    interventions["vi_"+i_var] = np.linspace(1.1, 5.2, slider_tick_num).tolist() 
                elif i_var == "tiredness":
                    interventions["vi_"+i_var] = np.linspace(2., 7., slider_tick_num).tolist() 
                else:
                    interventions["vi_"+i_var] = np.linspace(1.,10., slider_tick_num).tolist() 
                ## Shift
                is_s = simulate_shift_intervention({i_var:interventions["si_"+i_var]}, model, trace, sim_i_pp_samples, var_names = i_vars) 
                ## Variance
                iv_s = simulate_variance_intervention({i_var:interventions["vi_"+i_var]}, model, trace, sim_i_pp_samples, var_names = i_vars)
                for v in is_s:
                    if i_var not in is_samples:
                        is_samples[i_var] = np.expand_dims(is_s[v], axis=0)
                        iv_samples[i_var] = np.expand_dims(iv_s[v], axis=0)
                    else:
                        is_s[v] = np.expand_dims(is_s[v], axis=0)
                        is_samples[i_var] = np.concatenate((is_samples[i_var], is_s[v]), axis=0)
                        iv_s[v] = np.expand_dims(iv_s[v], axis=0)
                        iv_samples[i_var] = np.concatenate((iv_samples[i_var], iv_s[v]), axis=0)
        ## Add intervention values
        data.observed_data.update(xr.Dataset.from_dict({var:{"dims": (var[3:]+'_i_value'),"data": interventions[var]} for var in interventions}))    
        data.observed_data = data.observed_data.assign_coords({var+'_i_value':(var+'_i_value',np.arange(0,len(interventions["ai_"+var]),1)) for var in i_vars})
        ## Add pp samples after intervention
        coords = {}
        coords['var'] = [v for v in ia_s]
        for var in ia_samples:
            coords[var+'_i_value'] = np.arange(0,ia_samples[var].shape[1],1)
        for i,c in enumerate(data.posterior_predictive.coords):
            coords[c] = np.arange(0,ia_samples[var].shape[i+2],1)
        ##
        dims = {}
        for var in data.posterior_predictive.variables:
            if var not in coords:
                dims[var] = ['var']
                dims[var].append(var+'_i_value')
                dims[var].extend(list(data.posterior_predictive.variables[var].dims))
        ##
        data.add_groups({'atomic_intervention':ia_samples}, coords = coords, dims = dims)
        data.add_groups({'shift_intervention':is_samples}, coords = coords, dims = dims)
        data.add_groups({'variance_intervention':iv_samples}, coords = coords, dims = dims)
    # save data      
    arviz_to_json(data, fileName+'.npz')
    return data

def get_graph(header_js):
    graph = header_js["inference_data"]["sample_stats"]["attrs"]["graph"]
    return json.loads(graph.replace("'", "\""))
    
def get_causal_dag(graph):    
    deterministic = [i for i,v in graph.items() if v['type'] == 'deterministic']
    obs_vars = [i[:-1] for i,v in graph.items() if v['type'] == 'observed' and i[:-1] in deterministic]
    dag = {}
    for var in obs_vars:
        dag[var] = [p for p in graph[var+"_"]['parents'] if p in obs_vars]
    return dag

def get_var_dist_type(graph, var_name):
    """"
        Return any in {"Continuous","Discrete"}
    """
    if "type" in graph[var_name]["distribution"]:
        return graph[var_name]["distribution"]["type"]
    else:
        return ""
    
def get_var_dist(graph, var_name):
    if "dist" in graph[var_name]["distribution"]:
        return graph[var_name]["distribution"]["dist"]
    else:
        return graph[var_name]["type"]
    
def get_causal_inference_from_files(files):
    """
    Parameters:
    -----------
        files: List of file paths
        
    """
    ## Get intervention values per observed variable
    atomic_interventions = {}
    shift_interventions = {}
    variance_interventions = {}
    ##
    causal_inference = {}
    causal_inference['dags'] = {}
    causal_inference["vars"] = {}
    ##
    observations = {}
    for i,file in enumerate(files):
        inf_data = np.load(file)
        header_js = json.loads(inf_data['header.json'])
        graph = get_graph(header_js)
        dag = get_causal_dag(graph)
        ##
        causal_inference['dags'][i] = {}
        causal_inference['dags'][i]['dag'] = dag
        ## PP SAMPLES        
        causal_inference['dags'][i]['pp_samples'] = {} ## Dict <i_var>: numpy array of shape (chain, draw, obs_id) of posterior predictive samples
        ## INTERVENTIONS       
        causal_inference['dags'][i]['ia_samples'] = {}## Dict <i_var>: Dict <var>: numpy array of samples after intervention of shape (i_value, chain, draw, obs_id)
        causal_inference['dags'][i]['is_samples'] = {}
        causal_inference['dags'][i]['iv_samples'] = {}
        for i_var in dag:
            if i_var not in atomic_interventions:
                causal_inference["vars"][i_var] = {}
                causal_inference["vars"][i_var]["type"] = get_var_dist_type(graph, i_var+"_")
                atomic_interventions[i_var] = inf_data[header_js["inference_data"]["observed_data"]['vars']["ai_"+i_var]["array_name"]].tolist()
                causal_inference["vars"][i_var]["dist"] = get_var_dist(graph, i_var+"_")
                if causal_inference["vars"][i_var]["dist"] == "Normal":
                    shift_interventions[i_var] = inf_data[header_js["inference_data"]["observed_data"]['vars']["si_"+i_var]["array_name"]].tolist()
                    variance_interventions[i_var] = inf_data[header_js["inference_data"]["observed_data"]['vars']["vi_"+i_var]["array_name"]].tolist()
            ## pp_samples
            causal_inference['dags'][i]['pp_samples'][i_var] = inf_data[header_js["inference_data"]["posterior_predictive"]['vars'][i_var]["array_name"]]
            ## atomic
            causal_inference['dags'][i]['ia_samples'][i_var] = {}
            for coord_idx,var in enumerate(header_js["inference_data"]["atomic_intervention"]["coords"]["var"]):
                causal_inference['dags'][i]['ia_samples'][i_var][var] = inf_data[header_js["inference_data"]["atomic_intervention"]['vars'][i_var]["array_name"]][coord_idx]
            ## shift-variance
            if i_var in shift_interventions:
                causal_inference['dags'][i]['is_samples'][i_var] = {}
                causal_inference['dags'][i]['iv_samples'][i_var] = {}
                for coord_idx, var in enumerate(header_js["inference_data"]["shift_intervention"]["coords"]["var"]):
                    causal_inference['dags'][i]['is_samples'][i_var][var] = inf_data[header_js["inference_data"]["shift_intervention"]['vars'][i_var]["array_name"]][coord_idx]
                for coord_idx, var in enumerate(header_js["inference_data"]["variance_intervention"]["coords"]["var"]):
                    causal_inference['dags'][i]['iv_samples'][i_var][var] = inf_data[header_js["inference_data"]["variance_intervention"]['vars'][i_var]["array_name"]][coord_idx]
            ##
            if i_var not in observations:
                observations[i_var] = inf_data[header_js["inference_data"]["constant_data"]['vars'][i_var.upper()]["array_name"]]
    ## estimate var x_range across models
    estimate_x_range_across_models(observations, causal_inference)
    ## 
    return observations, atomic_interventions, shift_interventions, variance_interventions, causal_inference
            
        