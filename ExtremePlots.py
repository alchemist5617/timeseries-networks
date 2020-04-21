import numpy as np
import math
import pandas as pd
from pandas import Series
import random

from collections import OrderedDict
from netCDF4 import Dataset
from numpy import linspace
from numpy import meshgrid


from Data import Data
from datetime import datetime


import itertools
from collections import Counter
import pickle

from numpy import linalg as LA

import matplotlib.cm as cm

from scipy import stats

from matplotlib import animation

import scipy.stats as st



def spiParametersMle(x):
    i = x == 0
    params = st.gamma.fit(x[~i])
    q = len(x[i])/len(x)
    return(params, q)

def spiGeneratorMle(x):
    i = x == 0
    
    params = st.gamma.fit(x[~i])
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    G = st.gamma.cdf(x[~i], loc=loc, scale=scale, *arg)
    
    q = len(x[i])/len(x)
    probabilities = np.zeros(len(x))
    probabilities[i] = q
    probabilities[~i] = q + (1 - q) * G
    result = st.norm.ppf(probabilities)
    return(result)

def spiGeneratorParamMle(x, params, q):
    i = x == 0
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    G = st.gamma.cdf(x[~i], loc=loc, scale=scale, *arg)
    
    probabilities = np.zeros(len(x))
    probabilities[i] = q
    probabilities[~i] = q + (1 - q) * G
    result = st.norm.ppf(probabilities)
    return(result)    



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
class_dic = load_obj("class_dic")


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


def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation. 
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))


# In[7]:


level = 12
temporal_limits = {"time_min":datetime(1946, 1, 1, 0, 0),"time_max":datetime(2016, 1, 1, 0, 0) } 
spatial_limits = {"lon_min":-40,"lon_max":60,"lat_min":-40,"lat_max":40}

d = Data('precipitation.nc','precip',temporal_limits,missing_value=-9.969209968386869e+36)

result = d.get_data()
lon_list = d.get_lon_list()
lat_list = d.get_lat_list()
lon = d.get_lon()
lat = d.get_lat()


result = pd.DataFrame(result)

rolling_n = 3
f = 12
n = 30

RFThree = result.rolling(rolling_n).apply(sum)
RFThree = RFThree.iloc[rolling_n - 1:,:]

N = RFThree.shape[0]

d3 = N - (n*f + 1)

result_index = []
for k in range(d3):
    onset = k
    end = k + (n*f - (rolling_n - 1))
    
    a = RFThree.iloc[onset:end,:].values
    b = RFThree.iloc[end + (rolling_n - 1),:].values
    n_a = a.shape[1]
    index = np.zeros(n_a, dtype=bool)
    
    for i in range(n_a):
        x = a[:,i]
        params,q = spiParametersMle(x)
        r = spiGeneratorParamMle([b[i]], params, q)
        if r[0] < -1:
            index[i] = True

    result_index.append(index)

np.save("SPI3Index.npy",np.array(result_index))

