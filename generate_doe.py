import pandas as pd
import pyDOE
import numpy as np



def generate_doe (n_sample, doe_params, verbose = False, iterations=10000, numeric=False):

    myprint = print if verbose else lambda *args: None
    
    norm_doe = generate_raw_doe (n_sample, doe_params, verbose = verbose, iterations=iterations)
    myprint("correlation",norm_doe.corr())
    
   
    df=map_doe_to_labels (norm_doe, doe_params , numeric=False)
    
    return df, norm_doe

def generate_raw_doe (n_sample, doe_params, verbose = False, iterations=10000):

    myprint = print if verbose else lambda *args: None
    
    n_dim = doe_params.shape[1]
    myprint("ndim={}".format(n_dim))
    
    
    
    norm_doe = pd.DataFrame ( pyDOE.lhs( n_dim , samples= n_sample, criterion = "corr" , iterations=iterations), columns= doe_params.columns )
    

    
    myprint("correlation",norm_doe.corr())
    
   
    return norm_doe


def map_doe_to_labels (norm_doe, doe_params, numeric=False):

    
    possibilities_per_dim = doe_params.apply(lambda col:len(col.dropna().unique()))
    #myprint("possibilities_per_dim",possibilities_per_dim)
    
    df=pd.DataFrame()

    if numeric:
        for c in norm_doe:
            df[c] = pd.cut(norm_doe[c], bins = possibilities_per_dim[c])
           
        return    df.applymap(lambda x:x.right)     
    
    else:
        for c in norm_doe:
            df[c] = pd.cut(norm_doe[c], bins = possibilities_per_dim[c],  labels=doe_params[c].dropna())
        return df
    
def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm
    import seaborn as sns

    fig = plt.figure(figsize=(11,11))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('Spectral_r', 8)
    
    mat = df.corr().abs().values
    np.fill_diagonal(mat, 0)
    corr=pd.DataFrame(mat, index=df.columns, columns=df.columns)

    sns.heatmap(corr, cmap=cmap,
        xticklabels=corr.columns,
        yticklabels=corr.columns)
    
    

