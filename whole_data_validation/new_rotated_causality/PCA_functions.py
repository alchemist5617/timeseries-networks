"""Created on Thu Sep 19 10:27:27 2019

@author: M Noorbakhsh
"""

import numpy as np
from scipy import eye, asarray, dot, sum, diag
from scipy.linalg import svd
import random
import scipy.stats as st
import math
import pandas as pd
import pickle


def varimax(Phi, gamma = 1.0, q = 1000, tol = 1e-6):
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    return dot(Phi, R)

def orthomax(U, rtol = np.finfo(np.float32).eps ** 0.5, gamma = 1.0, maxiter = 1000):
    """
    Rotate the matrix U using a varimax scheme.  Maximum no of rotation is 1000 by default.
    The rotation is in place (the matrix U is overwritten with the result).  For optimal performance,
    the matrix U should be contiguous in memory.  The implementation is based on MATLAB docs & code,
    algorithm is due to DN Lawley and AE Maxwell.
    """
    n,m = U.shape
    Ur = U.copy(order = 'C')
    ColNorms = np.zeros((1, m))
    
    dsum = 0.0
    for indx in range(maxiter):
        old_dsum = dsum
        np.sum(Ur**2, axis = 0, out = ColNorms[0,:])
        C = n * Ur**3
        if gamma > 0.0:
            C -= gamma * Ur * ColNorms  # numpy will broadcast on rows
        L, d, Mt = svd(np.dot(Ur.T, C), False, True, True)
        R = np.dot(L, Mt)
        dsum = np.sum(d)
        np.dot(U, R, out = Ur)
        if abs(dsum - old_dsum) / dsum < rtol:
            break
        
    # flip signs of components, where max-abs in col is negative
    for i in range(m):
        if np.amax(Ur[:,i]) < -np.amin(Ur[:,i]):
            Ur[:,i] *= -1.0
            R[i,:] *= -1.0
            
    return Ur, R, indx

def uni_deseasonalize(ts,freq=12):
    ts = np.array(ts)
    N = len(ts)
    #averages = np.zeros((freq,n))
    temp = ts
    result = np.zeros((N))
    for j in range(freq):
        Idx = np.arange(j,N,freq)
        result[Idx] = (temp[Idx] - temp[Idx].mean())/temp[Idx].std()
    return(result) 


def deseasonalize(data,freq=12):
    """
    The shape of data should be (time, index) 
    """
    n  = data.shape[1]
    N  = data.shape[0]
    data_deseasonal = np.zeros(data.shape)
    for i in range(n):
        temp = np.copy(data[:,i])
        temp = np.ravel(temp)
        r = np.zeros((N))
        for j in range(freq):
            Idx = np.arange(j,N,freq)
            if temp[Idx].std() == 0:
                r[Idx] = 0
            else:
                r[Idx] = (temp[Idx] - temp[Idx].mean())/temp[Idx].std()
        data_deseasonal[:,i] = np.copy(r)
        
    return(data_deseasonal) 
    
def deseasonalize_avg_std(data,freq=12):
    """
    The shape of data should be (time, index) 
    """
    n  = data.shape[1]
    N  = data.shape[0]
    data_deseasonal = np.zeros(data.shape)
    Avgs = np.zeros((n,freq))
    Stds = np.zeros((n,freq))
    for i in range(n):
        temp = np.copy(data[:,i])
        temp = np.ravel(temp)
        r = np.zeros((N))
        for j in range(freq):
            Idx = np.arange(j,N,freq)
            if temp[Idx].std() == 0:
                r[Idx] = 0
            else:
                r[Idx] = (temp[Idx] - temp[Idx].mean())/temp[Idx].std()
            Avgs[i,j] = temp[Idx].mean()
            Stds[i,j] = temp[Idx].std()
        data_deseasonal[:,i] = np.copy(r)       
    return(data_deseasonal, Avgs, Stds) 
    
def deseasonalize_NoStd(data,freq=12):
    n  = data.shape[1]
    N  = data.shape[0]
    averages = np.zeros((freq,n))
    data_deseasonal = np.zeros(data.shape)
    for i in range(n):
        temp = data[:,i]
        result = np.zeros((N))
        for j in range(freq):
            Idx = np.arange(j,N,freq)
            averages[j,i] = temp[Idx].mean()
            result[Idx] = temp[Idx] - temp[Idx].mean()
        data_deseasonal[:,i] = result
    return(data_deseasonal,averages)

def index_finder(loading,col,percent = 0.98):
    values = loading[col].sort_values(ascending = False)
    s = 0
    i = 0
    while s < percent:
        s+=values[i]
        i = i+1
    Idx = values[:i].index
    
    return(values[:i],Idx)

def index_finder_percentile(loading,col,percentile = 0.9):
    threshold = loading[col].quantile(percentile)
    values = loading[col].values   
    Idx = np.where(values>=threshold)[0]
    return(values[values>=threshold],Idx)

def random_color(n):
    result = []
    for i in range(n):
        r = lambda: random.randint(0,255)
        result.append('#%02X%02X%02X' % (r(),r(),r()))
    return(result)

def random_color_1(n):

    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(n)]
    return(color)

def aggregation(d,level):
    r = []
    for z in range(0,len(d)-level,level):
        r.append(np.nansum(d[range(z,z+level)]))
    return(np.array(r))


def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [        
        #st.beta,st.chi2,
        #st.expon,st.exponnorm,
        #st.genextreme,st.gamma,st.gengamma,
        #st.halfnorm,st.invgamma,st.invgauss,
        #st.norm,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,
        #st.uniform
        st.norm, st.gamma, st.pearson3,st.invgauss
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
       # try:
        #    # Ignore warnings from data that can't be fit
         #   with warnings.catch_warnings():
          #      warnings.filterwarnings('ignore')

                # fit dist to data
        params = distribution.fit(data)

                # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
        pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
        sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                #try:
                 #   if ax:
        pd.Series(pdf, x).plot(ax=ax,legend = True, label = distribution.name)
                  #  end
               # except Exception:
                #    pass

                # identify if this distribution is better
        if best_sse > sse > 0:
            best_distribution = distribution
            best_params = params
            best_sse = sse

      #  except Exception:
     #       pass

    return (best_distribution.name, best_params)

def phase_averaging(data,freq = 12):
    N = len(data)
    temp = data
    result = np.zeros(N)
    averages = np.zeros(freq)
    for j in range(freq):
        Idx = np.arange(j,N,freq)
        averages[j] = temp[Idx].mean()
        result[Idx] = (temp[Idx] - temp[Idx].mean())/temp[Idx].std()
    return(result, averages)

def fuzzify(x):
  # Add some "measurement error"" to each data point
    zero_idx = x==0
    x[zero_idx]+=0.005*np.random.uniform(0,1,1)[0]
    x[~zero_idx]+=0.005*np.random.uniform(-1,1,1)[0]
    return(x)
    
#def transform(data):
#    n  = data.shape[1]
#    N  = data.shape[0]
#    data_transformed = np.zeros(data.shape)
#    for i in range(n):
#        x = fuzzify(pd.DataFrame(result[:,i]))[0].values
#        data_transformed[:,i], lambda_ = stats.boxcox(x)
#    return(data_transformed)

def best_fit_distribution1(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [        
        #st.beta,st.chi2,
        #st.expon,st.exponnorm,
        #st.genextreme,st.gamma,st.gengamma,
        #st.halfnorm,st.invgamma,st.invgauss,
        #st.norm,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,
        #st.uniform
        st.norm, st.gamma, st.pearson3, st.genextreme 
        #st.dweibull, st.invgauss, st.lognorm
        #, st.gumbel_l, st.gumbel_r
        #"norm", "gamma", "pearson3", "genextreme", 
        #"dweibull", "invgauss", "lognorm", "gumbel_l", "gumbel_r"
        
    ]
    DISTRIBUTIONS_NAMES = [        
        #st.beta,st.chi2,
        #st.expon,st.exponnorm,
        #st.genextreme,st.gamma,st.gengamma,
        #st.halfnorm,st.invgamma,st.invgauss,
        #st.norm,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,
        #st.uniform
        #st.norm, st.gamma, st.pearson3, st.genextreme, 
        #st.dweibull, st.invgauss, st.lognorm, st.gumbel_l, st.gumbel_r
        "norm", "gamma", "pearson3", "genextreme" 
        #"dweibull", "invgauss", "lognorm""
        
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution, dist_name in list(zip(DISTRIBUTIONS,DISTRIBUTIONS_NAMES)):

        # Try to fit the distribution
       # try:
        #    # Ignore warnings from data that can't be fit
         #   with warnings.catch_warnings():
          #      warnings.filterwarnings('ignore')

                # fit dist to data
        params = distribution.fit(data)
        #anderson = st.anderson(data, distribution)
        #kstest = st.kstest(data, distribution)
        stat, p = stats.kstest(x,dist_name, args=params)
        

                # Separate parts of parameters
            
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
        pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
        sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                #try:
                 #   if ax:
        #pd.Series(pdf, x).plot(ax=ax,legend = True, label = distribution.name)
                  #  end
               # except Exception:
                #    pass

                # identify if this distribution is better
        if best_sse > sse > 0:
            best_distribution = distribution
            best_params = params
            best_sse = sse
            best_p = p
      #  except Exception:
     #       pass

    return (best_distribution.name, best_params, best_p)

def phase_averaging(data,freq = 12):
    N = len(data)
    temp = data
    result = np.zeros(N)
    averages = np.zeros(freq)
    for j in range(freq):
        Idx = np.arange(j,N,freq)
        averages[j] = temp[Idx].mean()
        result[Idx] = (temp[Idx] - temp[Idx].mean())/temp[Idx].std()
    return(result, averages)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
#class_dic = load_obj("class_dic")

def neighbour_vote(df,lat,lon):
    shift = [[-0.25,0,0.25],[-0.25,0,0.25]]
    shift_list = []
    for element in itertools.product(*shift):
        shift_list.append(element)
    shift_list.remove((0,0))
    result = []
    for x,y in shift_list:
        if not df[(df.lats == (lat + y)) & (df.lons == (lon + x))].clusters.values.size == 0:           
            result.append(np.asscalar(df[(df.lats == (lat + y)) & (df.lons == (lon + x))].clusters.values))
        else:
            result.append(-1)   
            
    return(np.array(result))

def neighbour_average(pre_list,result,lon,lat):
    shift = [[-0.25,0,0.25],[-0.25,0,0.25]]
    shift_list = []
    for element in itertools.product(*shift):
        shift_list.append(element)
    shift_list.remove((0,0))
    r = []
    for x,y in shift_list:
        if (lon + x,lat + y) in pre_list:
            j = pre_list.index((lon + x,lat + y))
            r.append(result.iloc[:,j].values)
        #else:
         #   print("NO")
          #  print(lon + x,lat + y)
           # r.append(np.zeros(817))   
    r = np.array(r)        
    return(np.average(r, axis=0))
    #return(r)

def neighbour_vote_class(dic,lat,lon):
    shift = [[-0.25,0.25],[-0.25,0.25]]
    shift_list = []
    for element in itertools.product(*shift):
        shift_list.append(element)
    #shift_list.remove((0,0))
    result = []
    for x,y in shift_list:
        if lon > 180: lon -= 360
        if (lat + y, lon + x) in class_dic.keys():
            result.append(class_dic[(lat + y, lon + x)])   
    return(np.array(result))

    
def transform(data):
    n  = data.shape[1]
    N  = data.shape[0]
    data_transformed = np.zeros(data.shape)
    for i in range(n):
        x = fuzzify(pd.DataFrame(result[:,i]))[0].values
        data_transformed[:,i], lambda_ = stats.boxcox(x)
    return(data_transformed)

def neighbour_vote(df,lat,lon):
    shift = [[-0.25,0,0.25],[-0.25,0,0.25]]
    shift_list = []
    for element in itertools.product(*shift):
        shift_list.append(element)
    shift_list.remove((0,0))
    result = []
    for x,y in shift_list:
        if not df[(df.lats == (lat + y)) & (df.lons == (lon + x))].clusters.values.size == 0:           
            result.append(np.asscalar(df[(df.lats == (lat + y)) & (df.lons == (lon + x))].clusters.values))
        else:
            result.append(-1)   
            
        
    return(np.array(result))

def neighbour_vote_class(dic,lat,lon):
    shift = [[-0.25,0.25],[-0.25,0.25]]
    shift_list = []
    for element in itertools.product(*shift):
        shift_list.append(element)
    #shift_list.remove((0,0))
    result = []
    for x,y in shift_list:
        if lon > 180: lon -= 360
        if (lat + y, lon + x) in class_dic.keys():
            result.append(class_dic[(lat + y, lon + x)])   
    return(np.array(result))

