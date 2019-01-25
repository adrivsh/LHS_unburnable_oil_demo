import pandas as pd
import pyDOE

def generate_doe (n_sample, doe_params, verbose = False, iterations=10000):

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