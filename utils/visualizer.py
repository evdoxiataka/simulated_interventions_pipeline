import pandas as pd
import ipme
import seaborn as sns
from arviz_json import get_dag, arviz_to_json
import arviz as az
import numpy as np
import random

def scatter_matrix(samples, vars = [], shuffle_var = None):
    samples_fl = dict(zip(vars,[samples[var].flatten().tolist() if isinstance(samples[var], (np.ndarray, np.generic)) else samples[var] for var in vars if var in samples]))
    if shuffle_var and shuffle_var in samples_fl:
        random.shuffle(samples_fl[shuffle_var])
    sns.pairplot(pd.DataFrame(samples_fl), kind='reg',diag_kind = 'kde', corner=True, plot_kws={"scatter":True,"ci":None})
    
def comp_scatter_data_sim(data, samples, label, vars = []):
    data_fl = dict(zip(vars,[data[var].flatten().tolist() for var in vars if var in data]))
    samples_fl = dict(zip(vars,[samples[var].flatten().tolist() for var in vars if var in samples]))
    data_fl['type'] = ['obs']*len(data_fl[next(iter(data_fl))])
    samples_fl['type'] = [label]*len(samples_fl[next(iter(samples_fl))])
    df = pd.DataFrame(data_fl)
    df1 = df.append(pd.DataFrame(samples_fl), ignore_index=True)
    sns.pairplot(df1, hue="type", kind='reg', corner=True, plot_kws={"scatter":True,"ci":None})
    
def scatter_matrix_pp(samples, N_inter, N, pp_samples, vars = []):
    idx = [j for z in range (N_inter) for i in range(pp_samples) for j in range(N)]
    samples_fl = dict(zip(vars,[samples[var].flatten().tolist() if isinstance(samples[var], (np.ndarray, np.generic)) else samples[var] for var in vars if var in samples]))
    df = pd.DataFrame(samples_fl)
    df['obs_idx'] = idx
    sns.pairplot(df, kind='reg', hue = 'obs_idx',diag_kind = 'kde', corner=True,palette='Dark2', plot_kws={"scatter":True,"ci":None})
    
def scatter_matrix_pp_s(samples, N_inter, N, pp_samples, vars = []):
    idx = [i for z in range (N_inter) for i in range(pp_samples) for j in range(N)]
    samples_fl = dict(zip(vars,[samples[var].flatten().tolist() if isinstance(samples[var], (np.ndarray, np.generic)) else samples[var] for var in vars if var in samples]))
    df = pd.DataFrame(samples_fl)
    df['sample_idx'] = idx
    sns.pairplot(df, kind='reg', hue = 'sample_idx',diag_kind = 'kde', corner=True,palette='Dark2', plot_kws={"scatter":True,"ci":None})#, 
    
def ipp(trace, pp_samples, model, vars=[]):
    """
    
    """
    # will also capture all the sampler statistics
    data = az.from_pymc3(trace=trace, posterior_predictive=pp_samples)
    # insert dag into sampler stat attributes
    dag = get_dag(model)
    data.sample_stats.attrs["graph"] = str(dag)

    # save data     
    # data.to_netcdf(fileName+'.nc')
    arviz_to_json(data, 'infData'+'.npz')

    # plot IPP
    v = [var+'_' for var in vars]
    ipme.scatter_matrix('infData.npz',vars=v)