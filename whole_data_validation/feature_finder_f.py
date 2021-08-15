import numpy as np
import math
import pandas as pd
from Data import Data
#import reverse_geocoder as rg
from datetime import datetime
import Rung as rung
import PCA_functions as pf
from tigramite.tigramite.pcmci import PCMCI
from tigramite.tigramite.independence_tests import ParCorr
import tigramite.tigramite.data_processing as pp
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cluster import AgglomerativeClustering
from scipy import signal
from scipy import stats
import numpy.ma as ma
import pickle
from scipy import linalg
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def data_list_maker_V(data_sst, V, link):
    df = pd.DataFrame()
    for k in range(len(link)):
        df[str(k)] = time_series_maker_V(data_sst, V[:,link[k,0]-1])
        df[str(k)] = df[str(k)].shift(abs(link[k,1]))
    df = df.dropna()
    return(df)

def data_list_maker_cluster(data_sst, df_sst, link):
    df = pd.DataFrame()
    for k in range(len(link)):
        df[str(k)] = time_series_maker_cluster(data_sst, df_sst, link[k,0]-1)
        df[str(k)] = df[str(k)].shift(abs(link[k,1]))
    df = df.dropna()
    return(df)

def data_list_maker(data_sst, df_sst, V, link):
    df = pd.DataFrame()
    for k in range(len(link)):
        df_sst["pc"] = V[:,link[k,0]-1]
        df[str(k)] = time_series_maker(link[k,0]-1, df_sst, data_sst)
        df[str(k)] = df[str(k)].shift(abs(link[k,1]))
    df = df.dropna()
    return(df)
    
def shift_df(df, start_lag = 1, end_lag = 12):
    lags = np.arange(start_lag,end_lag + 1)
    df = df.assign(**{
    '{} (t-{})'.format(col, t): df[col].shift(t)
    for t in lags
    for col in df
    })
    df = df.dropna()
    return(df)

def number_non_random(data):
    """Assumes data of shape (N, T)"""   
    
    c = np.corrcoef(data)
    c = np.nan_to_num(c)
    eigenValues, eigenVectors = np.linalg.eig(c)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    return(np.count_nonzero(eigenValues > (1+math.sqrt(data.shape[0]/data.shape[1]))**2))

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return(np.array(diff))    

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
    
def drought_timeseries_class(file_name, index, start_year = 1922, end_year=2015, extremes_treshold = -1, base_year = 1922):
    start_index = (start_year - base_year) * 12
    end_index = start_index + (end_year - (start_year - 1))*12
    ET_gamma = np.load(file_name)
    ET_gamma = ET_gamma[:,index]
    N = ET_gamma.shape[0]
    count = []
    for i in range(N):
        count.append(np.count_nonzero(ET_gamma[i,:] <= extremes_treshold))
    count_detrend = signal.detrend(count[start_index:end_index])
    return(count[start_index:end_index], count_detrend)

def drought_timeseries(file_name, start_year = 1922, end_year=2015, extremes_treshold = -1, base_year = 1922):
    start_index = (start_year - base_year) * 12
    end_index = start_index + (end_year - (start_year - 1))*12
    ET_gamma = np.load(file_name)
    N = ET_gamma.shape[0]
    count = []
    for i in range(N):
        count.append(np.count_nonzero(ET_gamma[i,:] <= extremes_treshold))
    count_detrend = signal.detrend(count[start_index:end_index])
    return(count[start_index:end_index], count_detrend)
    
def spi_timeseries(file_name, start_year = 1922, end_year=2015, index = 0, base_year = 1922):
    start_index = (start_year - base_year) * 12
    end_index = start_index + (end_year - (start_year - 1))*12
    ET_gamma = np.load(file_name)
    return(ET_gamma[start_index:end_index,index])

def data_generator_avg_std(file_name, code, temporal_limits, avgs, stds, freq = 12, missing_value=-9.96921e+36):
    sst = Data(file_name,code,temporal_limits, missing_value= missing_value)

    data = sst.get_data()
    lat_sst_list = sst.get_lat_list()
    
    n  = data.shape[1]
    N  = data.shape[0]
    data_deseasonal = np.zeros(data.shape)
    for i in range(n):
        temp = np.copy(data[:,i])
        temp = np.ravel(temp)
        r = np.zeros((N))
        for j in range(freq):
            Idx = np.arange(j,N,freq)
            if stds[i,j] == 0:
                r[Idx] = 0
            else:
                r[Idx] = (temp[Idx] - avgs[i,j])/stds[i,j]
        data_deseasonal[:,i] = np.copy(r)
    
    #data_deseasonal = difference(data_deseasonal)  
    weights = np.sqrt(np.abs(np.cos(np.array(lat_sst_list)* math.pi/180)))
    for i in range(len(weights)):
        data_deseasonal[:,i] = weights[i] * data_deseasonal[:,i]
    
    return(data_deseasonal) 

def data_generator_soil_avg_std(file_name, code, temporal_limits, avgs, stds,INDEX, lat_sst_list, test_n=0, freq = 12, missing_value=0):
    sst = Data(file_name,code,temporal_limits, missing_value= missing_value)
    result = sst.get_data()
    if not test_n == 0: 
        data = result[-test_n*12:,INDEX]
 
    n  = data.shape[1]
    N  = data.shape[0]
    data_deseasonal = np.zeros(data.shape)
    for i in range(n):
        temp = np.copy(data[:,i])
        temp = np.ravel(temp)
        r = np.zeros((N))
        for j in range(freq):
            Idx = np.arange(j,N,freq)
            if stds[i,j] == 0:
                r[Idx] = 0
            else:
                r[Idx] = (temp[Idx] - avgs[i,j])/stds[i,j]
        data_deseasonal[:,i] = np.copy(r)
    
    #data_deseasonal = difference(data_deseasonal)  
    weights = np.sqrt(np.abs(np.cos(np.array(lat_sst_list)* math.pi/180)))
    for i in range(len(weights)):
        data_deseasonal[:,i] = weights[i] * data_deseasonal[:,i]
    
    return(data_deseasonal)


def data_generator_deseasonalized(file_name, code, temporal_limits, missing_value=-9.96921e+36):
    sst = Data(file_name,code,temporal_limits, missing_value= missing_value)

    result = sst.get_data()
    lat_sst_list = sst.get_lat_list() 
    result = pf.deseasonalize(np.array(result))
    weights = np.sqrt(np.abs(np.cos(np.array(lat_sst_list)* math.pi/180)))
    for i in range(len(weights)):
        result[:,i] = weights[i] * result[:,i]
    
#    return(result)
    
def data_generator(file_name, code, temporal_limits, missing_value=-9.96921e+36):
    sst = Data(file_name,code,temporal_limits, missing_value= missing_value)

    result = sst.get_data()
    
    return(result)
    

def PCA_computer(file_name, code, temporal_limits,n_components_sst=76, missing_value=-9.96921e+36):
    sst = Data(file_name,code,temporal_limits, missing_value= missing_value)

    result = sst.get_data()
    lon_sst_list = sst.get_lon_list()
    lat_sst_list = sst.get_lat_list()

    result_sst, avgs, stds = pf.deseasonalize_avg_std(np.array(result))
    result_sst = difference(result_sst)
    weights = np.sqrt(np.abs(np.cos(np.array(lat_sst_list)* math.pi/180)))
    for i in range(len(weights)):
        result_sst[:,i] = weights[i] * result_sst[:,i]

    #data_sst = pd.DataFrame(result_sst)
        
    V, U, S, ts, eig, explained, max_comps = rung.pca_svd(pd.DataFrame(result_sst) ,truncate_by='max_comps', max_comps=n_components_sst)
    
    df_sst = pd.DataFrame({"lons":lon_sst_list,"lats":lat_sst_list})

    lon_temp = df_sst["lons"].values
    lon_temp[lon_temp > 180] = lon_temp[lon_temp > 180] -360
    df_sst["lons"].vlues = lon_temp
     
    #ts = difference(ts)  

    return(result_sst, ts, V, df_sst, avgs, stds)
    
def PCA_computer_rotated(file_name, code, temporal_limits,n_components_sst=98, missing_value=-9.96921e+36):
    sst = Data(file_name,code,temporal_limits, missing_value= missing_value)

    result = sst.get_data()
    lon_sst_list = sst.get_lon_list()
    lat_sst_list = sst.get_lat_list()

    result_sst, avgs, stds = pf.deseasonalize_avg_std(np.array(result))
    result_sst = signal.detrend(result_sst, axis=0)
    weights = np.sqrt(np.abs(np.cos(np.array(lat_sst_list)* math.pi/180)))
    for i in range(len(weights)):
        result_sst[:,i] = weights[i] * result_sst[:,i]

    data_sst = pd.DataFrame(result_sst)
        
    V, U, S, ts, eig, explained, max_comps = rung.pca_svd(data_sst,truncate_by='max_comps', max_comps=n_components_sst)
    
   # loading_sst = pf.varimax(V, q=1000)
   # loading_sst = rung.svd_flip(loading_sst)
   # for z in range(loading_sst.shape[1]):
   #     loading_sst[:,z] = loading_sst[:,z] / np.linalg.norm(loading_sst[:,z])
   # 
   # V = loading_sst
    
    Vr, Rot = rung.varimax(V)
    Vr = rung.svd_flip(Vr)

    # Get explained variance of rotated components
    s2 = np.diag(S)**2 / (ts.shape[0] - 1.)

    # matrix with diagonal containing variances of rotated components
    S2r = np.dot(np.dot(np.transpose(Rot), np.matrix(np.diag(s2))), Rot)
    expvar = np.diag(S2r)

    sorted_expvar = np.sort(expvar)[::-1]
    # s_orig = ((Vt.shape[1] - 1) * s2) ** 0.5

    # reorder all elements according to explained variance (descending)
    nord = np.argsort(expvar)[::-1]
    Vr = Vr[:, nord]

    # Get time series of UNMASKED data
    comps_ts = np.matmul(np.array(data_sst),Vr)

    df_sst = pd.DataFrame({"lons":lon_sst_list,"lats":lat_sst_list})

    lon_temp = df_sst["lons"].values
    lon_temp[lon_temp > 180] = lon_temp[lon_temp > 180] -360
    df_sst["lons"].vlues = lon_temp
    
    return(result_sst, comps_ts, Vr, df_sst, avgs, stds)
    
def PCA_computer_rotated_locs(file_name, code, temporal_limits,n_components_sst=98, missing_value=-9.96921e+36):
    sst = Data(file_name,code,temporal_limits, missing_value= missing_value)

    result = sst.get_data()
    lon_sst_list = sst.get_lon_list()
    lat_sst_list = sst.get_lat_list()
    lon_sst = sst.get_lon()
    lat_sst = sst.get_lat()

    result_sst, avgs, stds = pf.deseasonalize_avg_std(np.array(result))
    result_sst = signal.detrend(result_sst, axis=0)
    weights = np.sqrt(np.abs(np.cos(np.array(lat_sst_list)* math.pi/180)))
    for i in range(len(weights)):
        result_sst[:,i] = weights[i] * result_sst[:,i]

    data_sst = pd.DataFrame(result_sst)
        
    V, U, S, ts, eig, explained, max_comps = rung.pca_svd(data_sst,truncate_by='max_comps', max_comps=n_components_sst)
        
    Vr, Rot = rung.varimax(V)
    Vr = rung.svd_flip(Vr)

    # Get explained variance of rotated components
    s2 = np.diag(S)**2 / (ts.shape[0] - 1.)

    # matrix with diagonal containing variances of rotated components
    S2r = np.dot(np.dot(np.transpose(Rot), np.matrix(np.diag(s2))), Rot)
    expvar = np.diag(S2r)

    sorted_expvar = np.sort(expvar)[::-1]
    # s_orig = ((Vt.shape[1] - 1) * s2) ** 0.5

    # reorder all elements according to explained variance (descending)
    nord = np.argsort(expvar)[::-1]
    Vr = Vr[:, nord]

    # Get time series of UNMASKED data
    comps_ts = np.matmul(np.array(data_sst),Vr)

    df_sst = pd.DataFrame({"lons":lon_sst_list,"lats":lat_sst_list})

    lon_temp = df_sst["lons"].values
    lon_temp[lon_temp > 180] = lon_temp[lon_temp > 180] -360
    df_sst["lons"].vlues = lon_temp
    
    return(result_sst, comps_ts, Vr, df_sst, lon_sst, lat_sst)


def PCA_soil_rotated(file_name, code, temporal_limits,n_components_sst=40, test_n = 0, missing_value=0):
    sst = Data(file_name,code,temporal_limits, missing_value= missing_value)

    result = sst.get_data()
    lon_sst_list = sst.get_lon_list()
    lat_sst_list = sst.get_lat_list()
    lon = sst.get_lon()
    lat = sst.get_lat()
    
    lons = np.arange(lon[0],lon[-1],2)
    lats = np.arange(lat[0],lat[-1],-2)

    INDEX = []
    for i in range(len(lon_sst_list)):
        if (lon_sst_list[i] in lons) and (lat_sst_list[i] in lats):
            INDEX.append(i)
    
    result = result[:,INDEX]
    lat_sst_list = np.array(lat_sst_list)[INDEX]
    lon_sst_list = np.array(lon_sst_list)[INDEX]

    if not test_n == 0:
        result = result[:-test_n*12,:]

    result_sst, avgs, stds = pf.deseasonalize_avg_std(np.array(result))
    result_sst = signal.detrend(result_sst, axis=0)
    weights = np.sqrt(np.abs(np.cos(np.array(lat_sst_list)* math.pi/180)))
    for i in range(len(weights)):
        result_sst[:,i] = weights[i] * result_sst[:,i]

    data_sst = pd.DataFrame(result_sst)
        
    V, U, S, ts, eig, explained, max_comps = rung.pca_svd(data_sst,truncate_by='max_comps', max_comps=n_components_sst)
        
    Vr, Rot = rung.varimax(V)
    Vr = rung.svd_flip(Vr)

    # Get explained variance of rotated components
    s2 = np.diag(S)**2 / (ts.shape[0] - 1.)

    # matrix with diagonal containing variances of rotated components
    S2r = np.dot(np.dot(np.transpose(Rot), np.matrix(np.diag(s2))), Rot)
    expvar = np.diag(S2r)

    sorted_expvar = np.sort(expvar)[::-1]
    # s_orig = ((Vt.shape[1] - 1) * s2) ** 0.5

    # reorder all elements according to explained variance (descending)
    nord = np.argsort(expvar)[::-1]
    Vr = Vr[:, nord]

    # Get time series of UNMASKED data
    comps_ts = np.matmul(np.array(data_sst),Vr)

    df_sst = pd.DataFrame({"lons":lon_sst_list,"lats":lat_sst_list})

    lon_temp = df_sst["lons"].values
    lon_temp[lon_temp > 180] = lon_temp[lon_temp > 180] -360
    df_sst["lons"].vlues = lon_temp
    
    return(result_sst, comps_ts, Vr, df_sst, avgs, stds, INDEX, lat_sst_list)


def PCMCI_generator(ts, count, tau_min = 0, tau_max = 12, alpha_level = 0.05, alpha_level_q = 0.01,save=False, multi_test = False, file_name="PCMCI_results"):
    result_extremes = np.array(count)
    result_extremes = result_extremes.reshape((-1,1))
    
    result_sst = np.array(ts)

    result = np.concatenate((result_extremes,result_sst), axis=1)
    result = np.array(result)

    dataframe = pp.DataFrame(result)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)

    results = pcmci.run_pcmci(tau_min=tau_min, tau_max=tau_max, pc_alpha=None)
    if save: 
        save_obj(results, file_name)    


    pq_matrix = results['p_matrix']
    val_matrix = results['val_matrix']
     
    if multi_test:
        q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')
        N = q_matrix.shape[0]
        link_dict = dict()
        for j in range(N):
            # Get the good links
            good_links = np.argwhere(q_matrix[:, j, 1:] <= alpha_level_q)
            # Build a dictionary from these links to their values
            links = {(i, -tau - 1): np.abs(val_matrix[i, j, abs(tau) + 1])
                     for i, tau in good_links}
            # Sort by value
            link_dict[j] = sorted(links, key=links.get, reverse=True)
            
        link = np.array(link_dict[0])
        if len(link) > 0:
            link = link[link[:,0] != 0,:]
    else:    
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
    
def time_series_maker(pc, df_sst, result, level = 99): 
    if np.abs(df_sst.pc.values.min()) > np.abs(df_sst.pc.values.max()):
        limit = np.percentile(df_sst.pc.values, 1 - level)
        df_sst.pc.values[df_sst.pc.values>=limit]=np.nan
    else:
        limit = np.percentile(df_sst.pc.values, level)
        df_sst.pc.values[df_sst.pc.values<=limit]=np.nan

    I = np.where(~np.isnan(df_sst.pc.values))[0]

    d = result[:,I].mean(axis=1)
    d = np.ravel(d)
    return(d)

def feature_score(base, feature,ratio= 0.8, n_estimators=100, max_depth=5):
    df = pd.concat([base, feature],axis=1)
    index = int(df.shape[0]*ratio)

    x_train, x_test = df.iloc[:index,1:], df.iloc[index:,1:]
    y_train, y_test = df.iloc[:index,0], df.iloc[index:,0]
    model = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return(mean_squared_error(y_pred, y_test))
    
def time_series_maker_V(data, V_value):
    return(np.ravel(np.matmul(data,V_value)))
    
def time_series_maker_cluster(result, df_sst, cluster):
    Idx = np.where((df_sst.clusters == cluster).values)[0]
    d = result[:,Idx].mean(axis=1)
    d = np.ravel(d)
    return(d)
 
def forward_feature_V(count, data_sst, link, V, df_sst, tau, ratio = 0.8, n_estimators=100, max_depth=5):
    result = []
    link_list = []
    start_lag = tau
    end_lag = tau + 11
    df = pd.DataFrame({"drought":count})
    
    df = shift_df(df, start_lag, end_lag)
    index = int(df.shape[0]*ratio)
    dim = df.shape[1]
    x_train, x_test = df.iloc[:index,1:dim], df.iloc[index:,1:dim]
    y_train, y_test = df.iloc[:index,0], df.iloc[index:,0]
    base_model = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators)
    base_model.fit(x_train, y_train)
    y_pred = base_model.predict(x_test)
    result.append(mean_squared_error(y_pred, y_test))
    
    if len(link) > 0:
        df = pd.DataFrame({"drought": count})
        lags = np.arange(start_lag,end_lag + 1)
        df = df.assign(**{
        '{} (t-{})'.format(col, t): df[col].shift(t)
        for t in lags
        for col in df
        })
        for k in range(len(link)):
    #        df_sst["pc"] = V[:,link[k,0]-1]
     #       df[str(k)] = time_series_maker(link[k,0]-1, df_sst, data_sst)
    
            df[str(k)] = time_series_maker_V(data_sst, V[:,link[k,0]-1])
            df[str(k)] = df[str(k)].shift(abs(link[k,1]))
        df = df.dropna()
        
        base = df.iloc[:,:13].copy()
        features = df.iloc[:,13:].copy()
        
        while features.shape[1]>0:
            min_mse = np.Inf
            min_index = 0
            for c in features.columns:
                mse = feature_score(base, features[c])
                if (result[-1] > mse) and (min_mse > mse):
                    min_mse = mse
                    min_index = c
            if isinstance(min_index, int): break
            result.append(min_mse)
            base = pd.concat([base, features[min_index]],axis=1)
            features = features.drop(min_index,1)
            link_list.append(link[int(min_index)])
            
        if len(link_list) > 0:        
            x_train = base.iloc[:,1:]
            y_train = base.iloc[:,0]
            model = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators)
            model.fit(x_train, y_train)
        else:
            model = base_model
            link_list = []
    else:
        model = base_model
        link_list = []

    return(np.array(link_list),base_model, model)
    
def forward_feature_hybrid(count, data_sst, link, V, data_soil, link_soil, V_soil, tau,  ratio = 0.8, n_estimators=100, max_depth=5):
    result = []
    link_list = []
    link_name = []
    start_lag = tau
    end_lag = tau + 11
    df = pd.DataFrame({"drought":count})
    
    df = shift_df(df, start_lag, end_lag)
    index = int(df.shape[0]*ratio)
    dim = df.shape[1]
    x_train, x_test = df.iloc[:index,1:dim], df.iloc[index:,1:dim]
    y_train, y_test = df.iloc[:index,0], df.iloc[index:,0]
    base_model = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators)
    base_model.fit(x_train, y_train)
    y_pred = base_model.predict(x_test)
    result.append(mean_squared_error(y_pred, y_test))
    
    df = pd.DataFrame({"drought": count})
    lags = np.arange(start_lag,end_lag + 1)
    df = df.assign(**{
    '{} (t-{})'.format(col, t): df[col].shift(t)
    for t in lags
    for col in df
    })
    for k in range(len(link)):
        df[str(k)] = time_series_maker_V(data_sst, V[:,link[k,0]-1])
        df[str(k)] = df[str(k)].shift(abs(link[k,1]))
        
    shift_r = len(link)
    for k in range(len(link_soil)):
        df[str(k+shift_r)] = time_series_maker_V(data_soil, V_soil[:,link_soil[k,0]-1])
        df[str(k+shift_r)] = df[str(k+shift_r)].shift(abs(link_soil[k,1]))
    
    df = df.dropna()
    
    base = df.iloc[:,:13].copy()
    features = df.iloc[:,13:].copy()
    
    while features.shape[1]>0:
        min_mse = np.Inf
        min_index = 0
        for c in features.columns:
            mse = feature_score(base, features[c])
            if (result[-1] > mse) and (min_mse > mse):
                min_mse = mse
                min_index = c
        if isinstance(min_index, int): break
        result.append(min_mse)
        base = pd.concat([base, features[min_index]],axis=1)
        features = features.drop(min_index,1)
        
        if int(min_index) < shift_r:
            link_list.append(link[int(min_index)])
            link_name.append('sst')
        else:
            link_list.append(link_soil[int(min_index)-shift_r])
            link_name.append('soil')
            
    if len(link_list) > 0:        
        x_train = base.iloc[:,1:]
        y_train = base.iloc[:,0]
        model = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators)
        model.fit(x_train, y_train)
    else:
        model = base_model
        link_list = []
    
    return(np.array(link_list),link_name,base_model, model)


def forward_feature_cluster(count, data_sst, link, df_sst, tau,  ratio = 0.8, n_estimators=100, max_depth=5):
    result = []
    link_list = []
    start_lag = tau
    end_lag = tau + 11
    df = pd.DataFrame({"drought":count})
    
    df = shift_df(df, start_lag, end_lag)
    index = int(df.shape[0]*ratio)
    dim = df.shape[1]
    x_train, x_test = df.iloc[:index,1:dim], df.iloc[index:,1:dim]
    y_train, y_test = df.iloc[:index,0], df.iloc[index:,0]
    base_model = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators)
    base_model.fit(x_train, y_train)
    y_pred = base_model.predict(x_test)
    result.append(mean_squared_error(y_pred, y_test))
    
    df = pd.DataFrame({"drought": count})
    lags = np.arange(start_lag,end_lag + 1)
    df = df.assign(**{
    '{} (t-{})'.format(col, t): df[col].shift(t)
    for t in lags
    for col in df
    })
    for k in range(len(link)):
        df[str(k)] = time_series_maker_cluster(data_sst, df_sst, link[k,0]-1)
        df[str(k)] = df[str(k)].shift(abs(link[k,1]))
    df = df.dropna()
    
    base = df.iloc[:,:13].copy()
    features = df.iloc[:,13:].copy()
    
    while features.shape[1]>0:
        min_mse = np.Inf
        min_index = 0
        for c in features.columns:
            mse = feature_score(base, features[c])
            if (result[-1] > mse) and (min_mse > mse):
                min_mse = mse
                min_index = c
        if isinstance(min_index, int): break
        result.append(min_mse)
        base = pd.concat([base, features[min_index]],axis=1)
        features = features.drop(min_index,1)
        link_list.append(link[int(min_index)])
            
    if len(link_list) > 0:        
        x_train = base.iloc[:,1:]
        y_train = base.iloc[:,0]
        model = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators)
        model.fit(x_train, y_train)
    else:
        model = base_model
        link_list = []
    
    return(np.array(link_list),base_model, model)   

def model_generator_V(count, data_sst, link, V, tau, ratio = 0.8, n_estimators=100, max_depth=5):
    
    start_lag = tau
    end_lag = tau+11
    
    df = pd.DataFrame({"drought":count})
    df = shift_df(df, start_lag, end_lag)
    x_train = df.iloc[:,1:]
    y_train = df.iloc[:,0]
    base_model = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators)
    base_model.fit(x_train, y_train)        
    
    if len(link) > 0:
        df = pd.DataFrame({"drought": count})
        lags = np.arange(start_lag,end_lag + 1)
        df = df.assign(**{
        '{} (t-{})'.format(col, t): df[col].shift(t)
        for t in lags
        for col in df
        })
        for k in range(len(link)):
            df[str(k)] = time_series_maker_V(data_sst, V[:,link[k,0]-1])
            df[str(k)] = df[str(k)].shift(abs(link[k,1]))
        df = df.dropna()
            
        x_train = df.iloc[:,1:]
        y_train = df.iloc[:,0]
        model = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators)
        model.fit(x_train, y_train)
            
        return(base_model, model)
    else:
        return(base_model, base_model)

def forward_feature(count, data_sst, df_sst, link, V, tau, ratio = 0.8, n_estimators=100, max_depth=5 ):
    result = []
    link_list = []
    start_lag = tau
    end_lag = tau + 12
    df = pd.DataFrame({"drought":count})
    
    df = shift_df(df, start_lag, end_lag)
    index = int(df.shape[0]*ratio)
    dim = df.shape[1]
    x_train, x_test = df.iloc[:index,1:dim], df.iloc[index:,1:dim]
    y_train, y_test = df.iloc[:index,0], df.iloc[index:,0]
    base_model = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators)
    base_model.fit(x_train, y_train)
    y_pred = base_model.predict(x_test)
    result.append(mean_squared_error(y_pred, y_test))
    
    df = pd.DataFrame({"drought":count})
    lags = np.arange(start_lag,end_lag + 1)
    df = df.assign(**{
    '{} (t-{})'.format(col, t): df[col].shift(t)
    for t in lags
    for col in df
    })
    for k in range(len(link)):
        df_sst["pc"] = V[:,link[k,0]-1]
        df[str(k)] = time_series_maker(link[k,0]-1, df_sst, data_sst)
        df[str(k)] = df[str(k)].shift(abs(link[k,1]))
    df = df.dropna()
    
    base = df.iloc[:,:14].copy()
    features = df.iloc[:,14:].copy()
    
    while features.shape[1]>0:
        min_mse = np.Inf
        min_index = 0
        for c in features.columns:
            mse = feature_score(base, features[c])
            if (result[-1] > mse) and (min_mse > mse):
                min_mse = mse
                min_index = c
        if isinstance(min_index, int): break
        result.append(min_mse)
        base = pd.concat([base, features[min_index]],axis=1)
        features = features.drop(min_index,1)
        link_list.append(link[int(min_index)])
        
        
    if len(link_list) > 0:        
        x_train = base.iloc[:,1:]
        y_train = base.iloc[:,0]
        model = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators)
        model.fit(x_train, y_train)
    else:
        model = base_model
        link_list = []
    
    return(np.array(link_list),base_model, model)       



def min_MSE_finder_cluster(count, result_sst, link, df_sst, ratio=0.8, tau=-1, n_estimators=100, max_depth=5):
    result = []
    link = link[link[:,1] <= tau]

    df = pd.DataFrame({"drought":count, "drought1":count})
    df.drought1 = df.drought1.shift(abs(tau))
    df = df.dropna()
    index = int(df.shape[0]*ratio)
    dim = df.shape[1]
    #index +=tau

    x_train, x_test = df.iloc[:index,1:dim], df.iloc[index:,1:dim]
    y_train, y_test = df.iloc[:index,0], df.iloc[index:,0]
    model = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    result.append(mean_squared_error(y_pred, y_test))


    for z in range(len(link)):
        df = pd.DataFrame({"drought":count, "drought1":count})
        df.drought1 = df.drought1.shift(abs(tau))
        for k in range(0,z+1):
                #df_sst["pc"] = V[:,link[k,0]-1]
                df[str(k)] = time_series_maker_cluster(result_sst, df_sst, link[k,0]-1 )
                df[str(k)] = df[str(k)].shift(abs(link[k,1]))
        df = df.dropna()
        index = int(df.shape[0]*ratio)
        #dim = df.shape[1]
        #index +=tau

        x_train, x_test = df.iloc[:index,1:], df.iloc[index:,1:]
        y_train, y_test = df.iloc[:index,0], df.iloc[index:,0]
        model = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        result.append(mean_squared_error(y_pred, y_test))
    return(result,link)
    
def best_link_finder_cluster(count, data_sst, link, df_sst, tau=-1, ratio=0.8, n_estimators=100, max_depth=5):

    result, link = min_MSE_finder_cluster(count, data_sst, link, df_sst, ratio, tau)

    overall_min_MSE = []
    overall_min_MSE.append(min(result))

    link_list = []
    diff = [x - result[i - 1] for i, x in enumerate(result)][1:]
    refined_index = np.array(diff) < 0
    while not all(refined_index):
        link_list.append(link)
        link = link[refined_index,:]
        result, link = min_MSE_finder_cluster(count, data_sst, link, df_sst, ratio, tau)
        overall_min_MSE.append(min(result))
        diff = [x - result[i - 1] for i, x in enumerate(result)][1:]
        refined_index = np.array(diff) < 0

    df = pd.DataFrame({"drought":count, "drought1":count})
    df.drought1 = df.drought1.shift(abs(tau))
    df = df.dropna()
    #index +=tau

    x_train =  df.iloc[:,1:].values
    y_train = df.iloc[:,0].values
    x_train  = x_train.reshape(-1, 1)
    y_train  = y_train.reshape(-1, 1)
    base_model = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators)
    base_model.fit(x_train, y_train)

    if len(link_list) > 0:
        best_index = np.where(np.array(overall_min_MSE) == np.array(overall_min_MSE).min())[0][-1]
        best_link = link_list[best_index-1]

        df = pd.DataFrame({"drought":count, "drought1":count})
        df.drought1 = df.drought1.shift(abs(tau))
        for k in range(len(best_link)):
           # df_sst["pc"] = V[:,best_link[k,0]-1]
            #df[str(best_link[k,0]-1)] = time_series_maker(best_link[k,0]-1, df_sst, data_sst)
            #df[str(best_link[k,0]-1)] = df[str(best_link[k,0]-1)].shift(abs(best_link[k,1]))
            df[str(k)] = time_series_maker_cluster(data_sst, df_sst, best_link[k,0]-1)
            df[str(k)] = df[str(k)].shift(abs(best_link[k,1]))
        df = df.dropna()
        #index +=tau

        x_train = df.iloc[:,1:]
        y_train = df.iloc[:,0]
        model = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators)
        model.fit(x_train, y_train)
    else:
        model = base_model
        best_link = []

    return(best_link, base_model, model)
    
def addtrend(initial, diff):
    original = []
    original.append(initial)
    for i in range(len(diff)):
        original.append(original[i] + diff[i].item())
    return(original)

def base_model_result(count, base_model, link, tau=-1):
    start_lag = tau
    end_lag = tau + 11
    df = pd.DataFrame({"drought":count})
    df = shift_df(df, start_lag, end_lag)

    x_test =  df.iloc[:,1:].values
    y_test = df.iloc[:,0].values

    y_pred = base_model.predict(x_test)
    
    return(y_pred, y_test)

def model_result(count, data_sst, link, df_sst, V, model, tau=1, n_estimators=100, max_depth=5):
    start_lag = tau
    end_lag = tau + 11  
    
    if len(link) > 0:
        df = pd.DataFrame({"drought":count})
        lags = np.arange(start_lag,end_lag + 1)
        df = df.assign(**{
        '{} (t-{})'.format(col, t): df[col].shift(t)
        for t in lags
        for col in df
        })
        for k in range(len(link)):
            df_sst["pc"] = V[:,link[k,0]-1]
            df[str(k)] = time_series_maker(link[k,0]-1, df_sst, data_sst)
            df[str(k)] = df[str(k)].shift(abs(link[k,1]))
        df = df.dropna()

        x_test = df.iloc[:,1:]
        y_test = df.iloc[:,0]

        y_pred = model.predict(x_test)
        return(y_pred, y_test)
    else:
        df = pd.DataFrame({"drought":count})
        df = shift_df(df, start_lag, end_lag)

        x_test =  df.iloc[:,1:].values
        y_test = df.iloc[:,0].values

        y_pred = model.predict(x_test)
        return(y_pred, y_test)
        
def model_result_hybrid(count, link, link_name, data_sst, df_sst, V, data_soil, df_soil, V_soil, model, tau=1, n_estimators=100, max_depth=5):
    if len(link) > 0:
        start_lag = tau
        end_lag = tau + 11

        df = pd.DataFrame({"drought":count})
        lags = np.arange(start_lag,end_lag + 1)
        df = df.assign(**{
        '{} (t-{})'.format(col, t): df[col].shift(t)
        for t in lags
        for col in df
        })
        for k in range(len(link)):
            if link_name[k] == 'sst':
                df_sst["pc"] = V[:,link[k,0]-1]
                df[str(k)] = time_series_maker(link[k,0]-1, df_sst, data_sst)
                df[str(k)] = df[str(k)].shift(abs(link[k,1]))
            else:
                df_soil["pc"] = V_soil[:,link[k,0]-1]
                df[str(k)] = time_series_maker(link[k,0]-1, df_soil, data_soil)
                df[str(k)] = df[str(k)].shift(abs(link[k,1]))
        df = df.dropna()

        x_test = df.iloc[:,1:]
        y_test = df.iloc[:,0]

        y_pred = model.predict(x_test)
        return(y_pred, y_test)
    else:
        return(np.nan, np.nan)

def model_result_V(count, data_sst, link, df_sst, V, model, tau=-1, n_estimators=100, max_depth=5): 
    if len(link) > 0:
        start_lag = tau
        end_lag = tau + 11

        df = pd.DataFrame({"drought":count})
        lags = np.arange(start_lag,end_lag + 1)
        df = df.assign(**{
        '{} (t-{})'.format(col, t): df[col].shift(t)
        for t in lags
        for col in df
        })
        for k in range(len(link)):
            df[str(k)] = time_series_maker_V(data_sst, V[:,link[k,0]-1])
            df[str(k)] = df[str(k)].shift(abs(link[k,1]))
        df = df.dropna()


        x_test = df.iloc[:,1:]
        y_test = df.iloc[:,0]

        y_pred = model.predict(x_test)
        return(y_pred, y_test)
    else:
        return(np.nan, np.nan)
        
def model_result_cluster(count, data_sst, link, df_sst, model, tau=1, n_estimators=100, max_depth=5):
    if len(link) > 0:
        start_lag = tau
        end_lag = tau + 11

        df = pd.DataFrame({"drought":count})
        lags = np.arange(start_lag,end_lag + 1)
        df = df.assign(**{
        '{} (t-{})'.format(col, t): df[col].shift(t)
        for t in lags
        for col in df
        })
        for k in range(len(link)):
            df[str(k)] = time_series_maker_cluster(data_sst, df_sst, link[k,0]-1)
            df[str(k)] = df[str(k)].shift(abs(link[k,1]))
        df = df.dropna()

        x_test = df.iloc[:,1:]
        y_test = df.iloc[:,0]

        y_pred = model.predict(x_test)
        return(y_pred, y_test)
    else:
        return(np.nan, np.nan)

def crosscorr(datax, datay, lag=1):   
    return(stats.pearsonr(datax[lag:], datay[:-lag]))
        
def corr_generator(ts, count, V, data_sst, tau_min = 1, tau_max = 12, level = 0.05, auto_corr = 0.8):
    result_extremes = np.array(count)
    result_extremes = result_extremes.reshape((-1,1))

    result_sst = np.array(ts)

    data = np.concatenate((result_extremes,result_sst), axis=1)
    data = np.array(data)

    N = data.shape[1]-1
    result = np.zeros((tau_max - tau_min + 1,N))

    for j in range(1,N+1):
        for i in range(tau_min,tau_max + 1):
            r, pvalue = crosscorr(data[:,0],data[:,j],lag=i)
            result[i-tau_min,j-1] = r if pvalue < level else 0
      
    result = np.abs(result)
    #limit = np.percentile(result, percentile)
    limit = 0
    Index = np.where(result > limit)
    link = np.array(list(zip((Index[1]+1),(Index[0] + tau_min)*(-1)))) 
    result = result[Index]
    link = link[(-result).argsort()]
    
    df = data_list_maker_V(data_sst, V, link)
    deleted_index = []
    componenets = set(link[:,0])
    for componenet in componenets:
        componenet_index = (link[:,0] == componenet)
        componenet_list = link[componenet_index]
        Index = componenet_index.nonzero()[0]
        sorted_index = np.argsort(componenet_list[:,1],axis=0)
        mx = ma.masked_array(componenet_list)
        for i in range(len(sorted_index)):
            if ma.is_masked(mx[sorted_index[i]]): continue
            for j in range(i+1,len(sorted_index)):
                if (not ma.is_masked(mx[sorted_index[i]]) and (df.iloc[:,Index[sorted_index[i]]].corr(df.iloc[:,Index[sorted_index[j]]]) > auto_corr)):
                    mx[sorted_index[j]] = ma.masked

        if not np.isscalar(mx.mask):
            deleted_index.extend(Index[mx.mask[:,0].nonzero()[0]])
    deleted_index = np.array(deleted_index)
    link = np.delete(link,deleted_index,axis=0)    
    
    return(link)

def calc_MI(x, y, bins = 10):
    c_xy = np.histogram2d(x, y, bins)[0]
    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
    #mi = 0.5 * g / c_xy.sum()
    mi = mutual_info_score(None, None, contingency=c_xy)
    return(mi, p)

def cross_MI(datax, datay, lag=1, bins = 10):   
    return(calc_MI(datax[lag:], datay[:-lag],bins))
        
def MI_generator(ts, count, V, data_sst, tau_min = 1, tau_max = 12, level = 0.05,auto_corr = 0.8, bins = 10):
    result_extremes = np.array(count)
    result_extremes = result_extremes.reshape((-1,1))

    result_sst = np.array(ts)

    data = np.concatenate((result_extremes,result_sst), axis=1)
    data = np.array(data)

    N = data.shape[1]-1
    result = np.zeros((tau_max - tau_min + 1,N))

    for j in range(1,N+1):
        for i in range(tau_min,tau_max + 1):
            MI, pvalue = cross_MI(data[:,0],data[:,j],lag=i, bins = bins)
            result[i-tau_min,j-1] = MI if pvalue < level else 0
      
    result = np.abs(result)
    #limit = np.percentile(result, percentile)
    limit = 0
    Index = np.where(result > limit)
    link = np.array(list(zip((Index[1]+1),(Index[0] + tau_min)*(-1)))) 
    result = result[Index]
    
    link = link[(-result).argsort()]       
    
    df = data_list_maker_V(data_sst, V, link)
    deleted_index = []
    componenets = set(link[:,0])
    for componenet in componenets:
        componenet_index = (link[:,0] == componenet)
        componenet_list = link[componenet_index]
        Index = componenet_index.nonzero()[0]
        sorted_index = np.argsort(componenet_list[:,1],axis=0)
        mx = ma.masked_array(componenet_list)
        for i in range(len(sorted_index)):
            if ma.is_masked(mx[sorted_index[i]]): continue
            for j in range(i+1,len(sorted_index)):
                if (not ma.is_masked(mx[sorted_index[i]]) and (df.iloc[:,Index[sorted_index[i]]].corr(df.iloc[:,Index[sorted_index[j]]]) > auto_corr)):
                    mx[sorted_index[j]] = ma.masked

        if not np.isscalar(mx.mask):
            deleted_index.extend(Index[mx.mask[:,0].nonzero()[0]])
    deleted_index = np.array(deleted_index)
    link = np.delete(link,deleted_index,axis=0) 

    return(link)

def corr_generator_cluster(ts, count, df_sst, data_sst, tau_min = 1, tau_max = 12, level = 0.05):
    result_extremes = np.array(count)
    result_extremes = result_extremes.reshape((-1,1))

    result_sst = np.array(ts)

    data = np.concatenate((result_extremes,result_sst), axis=1)
    data = np.array(data)

    N = data.shape[1]-1
    result = np.zeros((tau_max - tau_min + 1,N))

    for j in range(1,N+1):
        for i in range(tau_min,tau_max + 1):
            r, pvalue = crosscorr(data[:,0],data[:,j],lag=i)
            result[i-tau_min,j-1] = r if pvalue < level else 0
      
    result = np.abs(result)
    #limit = np.percentile(result, percentile)
    limit = 0
    Index = np.where(result > limit)
    link = np.array(list(zip((Index[1]+1),(Index[0] + tau_min)*(-1)))) 
    result = result[Index]
    link = link[(-result).argsort()]
    
    df = data_list_maker_cluster(data_sst, df_sst, link)
    deleted_index = []
    componenets = set(link[:,0])
    for componenet in componenets:
        componenet_index = (link[:,0] == componenet)
        componenet_list = link[componenet_index]
        Index = componenet_index.nonzero()[0]
        sorted_index = np.argsort(componenet_list[:,1],axis=0)
        mx = ma.masked_array(componenet_list)
        for i in range(len(sorted_index)):
            if ma.is_masked(mx[sorted_index[i]]): continue
            for j in range(i+1,len(sorted_index)):
                if (not ma.is_masked(mx[sorted_index[i]]) and (df.iloc[:,Index[sorted_index[i]]].corr(df.iloc[:,Index[sorted_index[j]]]) > 0.8)):
                    mx[sorted_index[j]] = ma.masked

        if not np.isscalar(mx.mask):
            deleted_index.extend(Index[mx.mask[:,0].nonzero()[0]])
    deleted_index = np.array(deleted_index)
    link = np.delete(link,deleted_index,axis=0)    
    
    return(link)

def clustering_computer(file_name, code, temporal_limits,n_components_sst=76, missing_value=-9.96921e+36):
    sst = Data(file_name,code,temporal_limits, missing_value= missing_value)

    result = sst.get_data()
    lon_sst_list = sst.get_lon_list()
    lat_sst_list = sst.get_lat_list()

    result_sst, avgs, stds = pf.deseasonalize_avg_std(np.array(result))
    result_sst = signal.detrend(result_sst, axis=0)
    weights = np.sqrt(np.abs(np.cos(np.array(lat_sst_list)* math.pi/180)))
    for i in range(len(weights)):
        result_sst[:,i] = weights[i] * result_sst[:,i]

    temp = np.array(result_sst)
    clustering = AgglomerativeClustering(n_clusters=n_components_sst).fit(np.transpose(temp))

    df_sst = pd.DataFrame({"lons":lon_sst_list,"lats":lat_sst_list,"clusters":clustering.labels_})

    lon_temp = df_sst["lons"].values
    lon_temp[lon_temp > 180] = lon_temp[lon_temp > 180] -360
    df_sst["lons"].vlues = lon_temp

    ts = np.zeros((result_sst.shape[0], n_components_sst))
    for i in range(n_components_sst):
        Idx = np.where((df_sst.clusters == i).values)[0]
        ts[:,i] = result_sst[:,Idx].mean(axis=1)

    return(result_sst, ts, df_sst, avgs, stds)


def base_model_result1(count, base_model, tau=-1):
    df = pd.DataFrame({"drought":count, "drought1":count})
    df.drought1 = df.drought1.shift(abs(tau))
    df = df.dropna()

    x_test =  df.iloc[:,1].values
    y_test = df.iloc[:,0].values
    x_test  = x_test.reshape(-1, 1)
    y_test  = y_test.reshape(-1, 1)

    y_pred = base_model.predict(x_test)
    return(mean_squared_error(y_pred, y_test))

def granger_generator(ts, count, V, data_sst, test_type = "all", tau_min = 1, tau_max = 12,level = 0.05, auto_corr = 0.8):
    result_extremes = np.array(count)
    result_extremes = result_extremes.reshape((-1,1))
    componenet = []
    lag = []
    result_sst = np.array(ts)
    
    for i in range(result_sst.shape[1]):
        sst = result_sst[:,i].reshape((-1,1))
        data = np.concatenate((result_extremes,sst), axis=1)
        data = np.array(data)
        df = pd.DataFrame(data)
        lag_range = range(tau_min, tau_max+1)
        r = grangercausalitytests(df,maxlag=lag_range,  verbose=False)
        p = np.zeros(4)
        for j in lag_range:
            p[0] = r[j][0]['lrtest'][1]
            p[1] = r[j][0]['params_ftest'][1]
            p[2] = r[j][0]['ssr_chi2test'][1]
            p[3] = r[j][0]['ssr_ftest'][1]
            
            
            if test_type == "all" and np.all(p < level):
                componenet.append(i+1)
                lag.append(-j)
            elif test_type == 'lrtest' and p[0] < level:
                componenet.append(i+1)
                lag.append(-j)
            elif test_type == 'params_ftest' and p[1] < level:
                componenet.append(i+1)
                lag.append(-j)
            elif test_type == 'ssr_chi2test' and p[2] < level:
                componenet.append(i+1)
                lag.append(-j)
            elif test_type == 'ssr_ftest' and p[3] < level:
                componenet.append(i+1)
                lag.append(-j)
            
    link = np.array(list(zip((componenet),(lag))))    
    
    df = data_list_maker_V(data_sst, V, link)
    deleted_index = []
    componenets = set(link[:,0])
    for componenet in componenets:
        componenet_index = (link[:,0] == componenet)
        componenet_list = link[componenet_index]
        Index = componenet_index.nonzero()[0]
        sorted_index = np.argsort(componenet_list[:,1],axis=0)
        mx = ma.masked_array(componenet_list)
        for i in range(len(sorted_index)):
            if ma.is_masked(mx[sorted_index[i]]): continue
            for j in range(i+1,len(sorted_index)):
                if (not ma.is_masked(mx[sorted_index[i]]) and (df.iloc[:,Index[sorted_index[i]]].corr(df.iloc[:,Index[sorted_index[j]]]) > auto_corr)):
                    mx[sorted_index[j]] = ma.masked

        if not np.isscalar(mx.mask):
            deleted_index.extend(Index[mx.mask[:,0].nonzero()[0]])
    deleted_index = np.array(deleted_index)
    link = np.delete(link,deleted_index,axis=0) 

    return(link)

def prod(iterable):
    """
    Product of a sequence of numbers.
    Faster than np.prod for short lists like array shapes, and does
    not overflow if using Python integers.
    """
    product = 1
    for x in iterable:
        product *= x
    return product

def detrend(data, axis=-1, type='linear', bp=0, overwrite_data=False):
    
    if type not in ['linear', 'l', 'constant', 'c']:
        raise ValueError("Trend type must be 'linear' or 'constant'.")
    data = np.asarray(data)
    dtype = data.dtype.char
    if dtype not in 'dfDF':
        dtype = 'd'
    if type in ['constant', 'c']:
        ret = data - np.expand_dims(np.mean(data, axis), axis)
        return ret
    else:
        dshape = data.shape
        N = dshape[axis]
        bp = np.sort(np.unique(np.r_[0, bp, N]))
        if np.any(bp > N):
            raise ValueError("Breakpoints must be less than length "
                             "of data along given axis.")
        Nreg = len(bp) - 1
        # Restructure data so that axis is along first dimension and
        #  all other dimensions are collapsed into second dimension
        rnk = len(dshape)
        if axis < 0:
            axis = axis + rnk
        newdims = np.r_[axis, 0:axis, axis + 1:rnk]
        newdata = np.reshape(np.transpose(data, tuple(newdims)),
                             (N, prod(dshape) // N))
        nd = newdata.copy()
        if not overwrite_data:
            newdata = newdata.copy()  # make sure we have a copy
        if newdata.dtype.char not in 'dfDF':
            newdata = newdata.astype(dtype)
        # Find leastsq fit and remove it for each piece
        for m in range(Nreg):
            Npts = bp[m + 1] - bp[m]
            A = np.ones((Npts, 2), dtype)
            A[:, 0] = np.cast[dtype](np.arange(1, Npts + 1) * 1.0 / Npts)
            sl = slice(bp[m], bp[m + 1])
            coef, resids, rank, s = linalg.lstsq(A, newdata[sl])
            newdata[sl] = newdata[sl] - np.dot(A, coef)
        # Put data back in original shape.
        tdshape = np.take(dshape, newdims, 0)
        ret = np.reshape(newdata, tuple(tdshape))
        vals = list(range(1, rnk))
        olddims = vals[:axis] + [0] + vals[axis:]
        ret = np.transpose(ret, tuple(olddims))
        return(ret, coef)

def model_generator_V_old(count, data_sst, link, V, tau, ratio = 0.8, n_estimators=100, max_depth=5):
    
    start_lag = tau
    end_lag = tau+12
    
    df = pd.DataFrame({"drought":count})
    df = ff.shift_df(df, start_lag, end_lag)
    x_train = df.iloc[:,1:]
    y_train = df.iloc[:,0]
    base_model = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators)
    base_model.fit(x_train, y_train)        
            
    df = pd.DataFrame({"drought": count})
    lags = np.arange(start_lag,end_lag + 1)
    df = df.assign(**{
    '{} (t-{})'.format(col, t): df[col].shift(t)
    for t in lags
    for col in df
    })
    for k in range(len(link)):
        df[str(k)] = ff.time_series_maker_V(data_sst, V[:,link[k,0]-1])
        df[str(k)] = df[str(k)].shift(abs(link[k,1]))
    df = df.dropna()
        
    x_train = df.iloc[:,1:]
    y_train = df.iloc[:,0]
    model = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators)
    model.fit(x_train, y_train)
        
    return(base_model, model)

def forward_feature_enso(count, enso, tau, ratio = 0.8, n_estimators=100, max_depth=5):
    result = []
    link_list = []
    start_lag = tau
    end_lag = tau + 11
    df = pd.DataFrame({"drought":count})

    df = shift_df(df, start_lag, end_lag)
    index = int(df.shape[0]*ratio)
    dim = df.shape[1]
    x_train, x_test = df.iloc[:index,1:dim], df.iloc[index:,1:dim]
    y_train, y_test = df.iloc[:index,0], df.iloc[index:,0]
    base_model = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators)
    base_model.fit(x_train, y_train)
    y_pred = base_model.predict(x_test)
    result.append(mean_squared_error(y_pred, y_test))

    df_count = pd.DataFrame({"drought": count})
    lags = np.arange(start_lag,end_lag + 1)
    df_count = df_count.assign(**{
    '{} (t-{})'.format(col, t): df_count[col].shift(t)
    for t in lags
    for col in df_count
    })

    df_enso = pd.DataFrame({"enso": enso})
    lags = np.arange(start_lag,end_lag + 1)
    df_enso = df_enso.assign(**{
    '{} (t-{})'.format(col, t): df_enso[col].shift(t)
    for t in lags
    for col in df_enso
    })

    df_enso = df_enso.drop(['enso'],1)

    df = pd.concat([df_count, df_enso],axis=1)
    df = df.dropna()

    base = df.iloc[:,:13].copy()
    features = df.iloc[:,13:].copy()

    while features.shape[1]>0:
        min_mse = np.Inf
        min_index = 0
        for c in features.columns:
            mse = feature_score(base, features[c])
            if (result[-1] > mse) and (min_mse > mse):
                min_mse = mse
                min_index = c
        if isinstance(min_index, int): break
        result.append(min_mse)
        base = pd.concat([base, features[min_index]],axis=1)
        features = features.drop(min_index,1)
        link_list.append(min_index)

    if len(link_list) > 0:        
        x_train = base.iloc[:,1:]
        y_train = base.iloc[:,0]
        model = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators)
        model.fit(x_train, y_train)
    else:
        model = base_model
        link_list = []
    
    return(np.array(link_list),base_model, model)
    
    
def model_result_enso(count, enso, link, model, tau=1, n_estimators=100, max_depth=5):
    if len(link) > 0:
        start_lag = tau
        end_lag = tau + 11

        df = pd.DataFrame({"drought":count})
        lags = np.arange(start_lag,end_lag + 1)
        df = df.assign(**{
        '{} (t-{})'.format(col, t): df[col].shift(t)
        for t in lags
        for col in df
        })
        
        df_enso = pd.DataFrame({"enso": enso})
        lags = np.arange(start_lag,end_lag + 1)
        df_enso = df_enso.assign(**{
        '{} (t-{})'.format(col, t): df_enso[col].shift(t)
        for t in lags
        for col in df_enso
        })

        df_enso = df_enso.drop(['enso'],1)
        
        for k in range(len(link)):
            df = pd.concat([df, df_enso[link[k]]],axis=1) 
        df = df.dropna()

        x_test = df.iloc[:,1:]
        y_test = df.iloc[:,0]

        y_pred = model.predict(x_test)
        return(y_pred, y_test)
    else:
        return(np.nan, np.nan)
def timeseries_enso(file_name, start_year = 1950, end_year=2015, base_year = 1948):
    start_index = (start_year - base_year) * 12   
    end_index = start_index + (end_year - (start_year - 1))*12
    data = np.load(file_name)
    return(data[start_index:end_index])
