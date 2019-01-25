import pandas as pd
import pyDOE
import numpy as np

def generate_doe_ (n_sample, doe_params, verbose = False, iterations=10000):

    myprint = print if verbose else lambda *args: None
    
    n_dim = doe_params.shape[1]
    myprint("ndim={}".format(n_dim))
    
    possibilities_per_dim = doe_params.apply(lambda col:len(col.dropna().unique()))
    myprint("possibilities_per_dim",possibilities_per_dim)
    
    norm_doe = pd.DataFrame ( pyDOE.lhs( n_dim , samples= n_sample, criterion = "corr" , iterations=iterations), columns= doe_params.columns )
    myprint("correlation",norm_doe.corr())
    
    
    df=pd.DataFrame()
    for c in norm_doe:
        df[c] = pd.cut(norm_doe[c], bins = possibilities_per_dim[c],  labels=doe_params[c].dropna())
    
    return df

def generate_doe (n_sample, doe_params, verbose = False, iterations=10000):

    myprint = print if verbose else lambda *args: None
    
    norm_doe = generate_raw_doe (n_sample, doe_params, verbose = verbose, iterations=iterations)
    myprint("correlation",norm_doe.corr())
    
    possibilities_per_dim = doe_params.apply(lambda col:len(col.dropna().unique()))
    myprint("possibilities_per_dim",possibilities_per_dim)
    
    df=map_doe_to_labels (norm_doe, doe_params, possibilities_per_dim)
    
    return df, norm_doe

def generate_raw_doe (n_sample, doe_params, verbose = False, iterations=10000):

    myprint = print if verbose else lambda *args: None
    
    n_dim = doe_params.shape[1]
    myprint("ndim={}".format(n_dim))
    
    
    
    norm_doe = pd.DataFrame ( pyDOE.lhs( n_dim , samples= n_sample, criterion = "corr" , iterations=iterations), columns= doe_params.columns )
    

    
    myprint("correlation",norm_doe.corr())
    
   
    return norm_doe


def map_doe_to_labels (norm_doe, doe_params, possibilities_per_dim):

    df=pd.DataFrame()
    for c in norm_doe:
        df[c] = pd.cut(norm_doe[c], bins = possibilities_per_dim[c],  labels=doe_params[c].dropna())
    
    return df
    
def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('Spectral_r', 5)
    
    mat = df.corr().abs().values
    np.fill_diagonal(mat, 0)
    
    cax = ax1.imshow(mat, interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    labels=df.columns
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax)
    plt.show()

