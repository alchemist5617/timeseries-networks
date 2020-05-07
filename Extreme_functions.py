#from statsmodels.tsa.arima_process import ArmaProcess 
#from statsmodels.tsa.stattools import pacf, acf
#from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from pandas import Series
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import eye, asarray, dot, sum, diag
from scipy.linalg import svd
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
from matplotlib import cm
from colorspacious import cspace_converter
from collections import OrderedDict
from netCDF4 import Dataset
from numpy import linspace
from numpy import meshgrid
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import grangercausalitytests

import PCA_functions as pf


from statsmodels.tsa.stattools import adfuller
from Data import Data
from datetime import datetime

from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.cluster import AgglomerativeClustering, DBSCAN

import itertools
from collections import Counter
import pickle

from numpy import linalg as LA

import matplotlib.cm as cm

from scipy.special import inv_boxcox
from scipy import stats

from matplotlib import animation

import scipy.stats as st

import reverse_geocoder as rg

from pandas_datareader import wb

level = 12
temporal_limits = {"time_min":datetime(1946, 1, 1, 0, 0),"time_max":datetime(2016, 1, 1, 0, 0) } 
spatial_limits = {"lon_min":-40,"lon_max":60,"lat_min":-40,"lat_max":40}


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

def extreme_plot_anamoly(cluster_id, rolling_n =12, extreme_type = "dry",n_components = 6 ):
    f = 12
    n = 30

    d = Data('../precipitation.nc','precip',temporal_limits,missing_value=-9.969209968386869e+36)

    result = d.get_data()
    lon_list = d.get_lon_list()
    lat_list = d.get_lat_list()
    lon = d.get_lon()
    lat = d.get_lat()

    result = pd.DataFrame(result)
    result = pf.deseasonalize(np.array(result))
    #result = np.array(result)

    temp = np.array(result)
    clustering = AgglomerativeClustering(n_clusters=n_components).fit(np.transpose(temp))

    df = pd.DataFrame({"lons":lon_list,"lats":lat_list,"clusters":clustering.labels_})

    lon_temp = df["lons"].values
    lon_temp[lon_temp > 180] = lon_temp[lon_temp > 180] -360
    df["lons"] = lon_temp

    clusters = clustering.labels_

    Idx = np.where((df.clusters == cluster_id).values)[0]

    d = Data('../precipitation.nc','precip',temporal_limits,missing_value=-9.969209968386869e+36)

    result = d.get_data()

    r = result[:,Idx]
    r = pd.DataFrame(r)


    RFThree = r.rolling(rolling_n).apply(sum)
    RFThree = RFThree.iloc[rolling_n - 1:,:]

    N = RFThree.shape[0]

    d3 = N - (n*f + 1)    
    
    result_index = []
    for k in range(d3):
        onset = k
        end = k + (n*f - (rolling_n - 1))

        r,avgs = deseasonalize_NoStd(RFThree.iloc[onset:end,:].values)
        #temp = RFThree.iloc[onset:end,:] - RFThree.iloc[onset:end,:].mean()
        r = pd.DataFrame(r)
        if extreme_type == "wet":
            a = r.quantile(0.95).values
            b = RFThree.iloc[end + (rolling_n - 1),:].values - avgs[(end + (rolling_n - 1)) % 12,:]
            index = np.where(np.greater(b,a))[0]
        else:            
            a = r.quantile(0.05).values
            b = RFThree.iloc[end + (rolling_n - 1),:].values - avgs[(end + (rolling_n - 1)) % 12,:]
            #b = RFThree.iloc[end + (rolling_n - 1) ,:].values
            index = np.where(np.greater(a,b))[0]
        result_index.append(index)

               
    number_cases = []
    for i in range(len(result_index)):
        number_cases.append(len(result_index[i]))
        
    
    if rolling_n == 12:
        start_date = "19761201"
    else:
        start_date = "19760{}01".format(rolling_n)
    df_oni = pd.DataFrame(number_cases,
                      columns=["number"],
                      index=pd.date_range(start_date, periods=len(number_cases), freq='MS'))

    oni = pd.read_csv("ONI.csv")
    
    if rolling_n == 12:
        oni_new = oni.iloc[323:792,:]
    elif rolling_n == 6:
        oni_new = oni.iloc[317:792,:]
    else:
        oni_new = oni.iloc[314:792,:]

    df_oni["oni"] = oni_new.iloc[:,5].values

    normalized_df=(df_oni-df_oni.mean())/df_oni.std()
    
    #title = "Cluster {} Cumulative anomaly rainfall {} {} extremes".format(cluster_id,rolling_n, extreme_type)
    #plot = normalized_df.plot(figsize=(20,10), title = title)
    #fig = plot.get_figure()
    #filename = "../plots/extreme_plot_anomaly__{}_months_cluster_{}_{}.png".format(rolling_n,cluster_id, extreme_type)
    #correlation = normalized_df.corr()
    #corr_text = "Correlation is {:.3f}".format(correlation.iloc[0,1])
    #fig.text(0.1,0.9,corr_text,fontsize=18)
    #fig.savefig(filename)
    return(df_oni)

def extreme_plot(cluster_id, rolling_n =12, extreme_type = "dry", n_components = 6):
    f = 12
    n = 30

    d = Data('../precipitation.nc','precip',temporal_limits,missing_value=-9.969209968386869e+36)

    result = d.get_data()
    lon_list = d.get_lon_list()
    lat_list = d.get_lat_list()
    lon = d.get_lon()
    lat = d.get_lat()

    result = pd.DataFrame(result)
    result = pf.deseasonalize(np.array(result))
    #result = np.array(result)

    temp = np.array(result)
    clustering = AgglomerativeClustering(n_clusters=n_components).fit(np.transpose(temp))

    df = pd.DataFrame({"lons":lon_list,"lats":lat_list,"clusters":clustering.labels_})

    lon_temp = df["lons"].values
    lon_temp[lon_temp > 180] = lon_temp[lon_temp > 180] -360
    df["lons"] = lon_temp

    clusters = clustering.labels_

    Idx = np.where((df.clusters == cluster_id).values)[0]

    d = Data('../precipitation.nc','precip',temporal_limits,missing_value=-9.969209968386869e+36)

    result = d.get_data()

    r = result[:,Idx]
    r = pd.DataFrame(r)


    RFThree = r.rolling(rolling_n).apply(sum)
    RFThree = RFThree.iloc[rolling_n - 1:,:]

    N = RFThree.shape[0]

    d3 = N - (n*f + 1)

    result_index = []
    for k in range(d3):
        onset = k
        end = k + (n*f - (rolling_n - 1))
        
        if extreme_type == "wet":
            a = RFThree.iloc[onset:end,:].quantile(0.95).values
            b = RFThree.iloc[end + (rolling_n - 1),:].values
            index = np.where(np.greater(b,a))[0]
            
        else:
            a = RFThree.iloc[onset:end,:].quantile(0.05).values
            b = RFThree.iloc[end + (rolling_n - 1),:].values
            index = np.where(np.greater(a,b))[0]

        result_index.append(index)

    number_cases = []
    for i in range(len(result_index)):
        number_cases.append(len(result_index[i]))
        
    
    if rolling_n == 12:
        start_date = "19761201"
    else:
        start_date = "19760{}01".format(rolling_n)
    df_oni = pd.DataFrame(number_cases,
                      columns=["number"],
                      index=pd.date_range(start_date, periods=len(number_cases), freq='MS'))

    oni = pd.read_csv("ONI.csv")
    
    if rolling_n == 12:
        oni_new = oni.iloc[323:792,:]
    elif rolling_n == 6:
        oni_new = oni.iloc[317:792,:]
    else:
        oni_new = oni.iloc[314:792,:]

    df_oni["oni"] = oni_new.iloc[:,5].values

    normalized_df=(df_oni-df_oni.mean())/df_oni.std()
    
    #title = "Cluster {} Cumulative rainfall {} months {} extremes".format(cluster_id,rolling_n, extreme_type)
    #plot = normalized_df.plot(figsize=(20,10), title = title)
    #fig = plot.get_figure()
    #filename = "../plots/extreme_plot_{}_months_cluster_{}_{}.png".format(rolling_n,cluster_id, extreme_type)
    #correlation = normalized_df.corr()
    #corr_text = "Correlation is {:.3f}".format(correlation.iloc[0,1])
    #fig.text(0.1,0.9,corr_text,fontsize=18)
    #fig.savefig(filename)
    return(df_oni)




