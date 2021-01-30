import numpy as np
import math
import pandas as pd
from Data import Data
import reverse_geocoder as rg
from datetime import datetime
import Rung as rung
import PCA_functions as pf
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, CMIknn
import tigramite.data_processing as pp

#from scipy.stats import norm

def number_non_random(data):
    """Assumes data of shape (N, T)"""   
    
    c = np.corrcoef(data)
    c = np.nan_to_num(c)
    eigenValues, eigenVectors = np.linalg.eig(c)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    return(np.count_nonzero(eigenValues > (1+math.sqrt(data.shape[0]/data.shape[1]))**2))
    
def data_finder(country_code, file_name = '../../nc/precip.mon.total.2.5x2.5.v2018.nc', temporal_limits= {"time_min":datetime(1891, 1, 1, 0, 0),"time_max":datetime(2015, 12, 1, 0, 0) } ):
    
    d = Data(file_name,'precip',temporal_limits,missing_value=-9.96921e+36)

    result = d.get_data()
    lon_list = d.get_lon_list()
    lat_list = d.get_lat_list()
    
    coordinates = list(zip(lat_list,lon_list))
    dic = rg.search(coordinates)
    country = []
    for i in range(len(dic)):
        country.append(dic[i].get('cc'))

    ET_index = np.where(np.array(country)== country_code)[0]
    ET_data = result[:,ET_index]
    return(ET_data)
    
def drought_timeseries(file_name, start_year = 1922, end_year=2015, extremes_treshold = -1, base_year = 1922):
    start_index = (start_year - base_year) * 12
    end_index = start_index + (end_year - (start_year - 1))*12
    ET_gamma = np.load(file_name)
    N = ET_gamma.shape[0]
    count = []
    for i in range(N):
        count.append(np.count_nonzero(ET_gamma[i,:] <= extremes_treshold))
    return(count[start_index:end_index])
    
def PCA_computer(file_name, code, temporal_limits, missing_value):
    sst = Data(file_name,code,temporal_limits, missing_value= missing_value)

    result = sst.get_data()
    lon_sst_list = sst.get_lon_list()
    lat_sst_list = sst.get_lat_list()

    result_sst = pf.deseasonalize(np.array(result))
    weights = np.sqrt(np.abs(np.cos(np.array(lat_sst_list)* math.pi/180)))
    for i in range(len(weights)):
        result_sst[:,i] = weights[i] * result_sst[:,i]

    n_components_sst = number_non_random(np.transpose(result_sst))
    data_sst = pd.DataFrame(result_sst)
        
    V, U, S, ts, eig, explained, max_comps = rung.pca_svd(data_sst,truncate_by='max_comps', max_comps=n_components_sst)
    
    df_sst = pd.DataFrame({"lons":lon_sst_list,"lats":lat_sst_list})

    lon_temp = df_sst["lons"].values
    lon_temp[lon_temp > 180] = lon_temp[lon_temp > 180] -360
    df_sst["lons"].vlues = lon_temp
    
    return(ts, result, df_sst)
    
def PCMCI_generator(ts, count):
    result_extremes = np.array(count)
    result_extremes = result_extremes.reshape((-1,1))
    
    result_sst = np.array(ts)

    result = np.concatenate((result_extremes,result_sst), axis=1)
    result = np.array(result)

    dataframe = pp.DataFrame(result)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)

    results = pcmci.run_pcmci(tau_max=12, pc_alpha=None)
    
    pq_matrix = results['p_matrix']
    val_matrix = results['val_matrix']
    alpha_level = 0.05
    N = pq_matrix.shape[0]

    link_dict = dict()
    for j in range(N):
        # Get the good links
        good_links = np.argwhere(pq_matrix[:, j, 1:] <= alpha_level)
        # Build a dictionary from these links to their values
        links = {(i, -tau - 1): np.abs(val_matrix[i, j, abs(tau) + 1])
                 for i, tau in good_links}
        # Sort by value
        link_dict[j] = sorted(links, key=links.get, reverse=True)
        
    link = np.array(link_dict[0])
    link = link[link[:,0] != 0,:]
    return(link)
    
def time_series_maker(pc, df_sst, result, level = 95): 
    if np.abs(df_sst.pc.values.min()) > np.abs(df_sst.pc.values.max()):
        limit = np.percentile(df_sst.pc.values, 1 - level)
        df_sst.pc.values[df_sst.pc.values>=limit]=np.nan
    else:
        limit = np.percentile(df_sst.pc.values, level)
        df_sst.pc.values[df_sst.pc.values<=limit]=np.nan
    
    I = np.where(~np.isnan(df_sst.pc.values))[0]
    d = result[:,I].mean(axis=1)
    d = np.ravel(d)
    d = np.reshape(d,(-1,1))
    d = pf.deseasonalize(d)
    d = np.ravel(d)
    return(d)
    