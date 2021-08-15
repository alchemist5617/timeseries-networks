import numpy as np
import feature_finder_f as ff
from datetime import datetime
from sklearn.metrics import mean_squared_error
import scipy.stats as st

#train_start = np.arange(1976,1977,5)
#train_end = np.arange(1955,2006,5)
#validation_end = np.arange(2010,2011,5)
#test_start = np.arange(2011,2012,5)
#test_end = np.arange(2015,2016,5)

step = 1
train_start = np.arange(1948,1977,step)
#train_end = np.arange(1955,2006,step)
validation_end = np.arange(1982,2011,step)
test_start = np.arange(1983,2012,step)
test_end = np.arange(1987,2016,step)

n_components_sst = np.load("sst_number.npy")
f = 12
taus = np.arange(1,13,1)
cc = 'ET'

for tau in taus:
    base_result = []
    model_results= []
    base_p = []
    model_p = []
    for ijz in range(len(train_start)):
        temporal_limits = {"time_min":datetime(train_start[ijz], 1, 1, 0, 0),"time_max":datetime(validation_end[ijz], 12, 1, 0, 0)}
        original_count, count = ff.drought_timeseries("../{}_gamma_18912015_{}_old.npy".format(cc,f),train_start[ijz],validation_end[ijz])
        data_sst, ts, V, df_sst, avg, std = ff.PCA_computer_rotated('../sst.mnmean.nc', "sst",temporal_limits, n_components_sst[ijz], -9.96921e+36)

        link = np.load("./link_multi/pcmci_{}_{}_{}_{}_{}.npy".format(f,train_start[ijz],validation_end[ijz],tau,n_components_sst[ijz]))
    
        base_model, model = ff.model_generator_V(count, data_sst, link, V, tau, ratio = 0.8, n_estimators=100, max_depth=5)
   
        temporal_limits = {"time_min":datetime(test_start[ijz], 1, 1, 0, 0),"time_max":datetime(test_end[ijz], 12, 1, 0, 0)}
        data_sst = ff.data_generator_avg_std('../sst.mnmean.nc', "sst",temporal_limits, avg, std, 12, -9.96921e+36)
        original_count_test, count_test = ff.drought_timeseries("../{}_gamma_18912015_{}_old.npy".format(cc,f),test_start[ijz],test_end[ijz])
        
        y_pred_base, y_test_base = ff.base_model_result(original_count_test, base_model, link, tau)
        y_pred, y_test = ff.model_result(original_count_test, data_sst, link, df_sst, V, model,tau)
        

        base_result.append(mean_squared_error(y_pred_base, y_test_base))
        base_p.append(st.pearsonr(y_pred_base, y_test_base)[0])
        
        if isinstance(y_pred, float) or isinstance(y_test, float):
            model_results.append(mean_squared_error(y_pred_base, y_test_base))
            model_p.append(st.pearsonr(y_pred_base, y_test_base)[0])
        else:
            model_results.append(mean_squared_error(y_pred, y_test))
            model_p.append(st.pearsonr(y_pred, y_test)[0])
        
 
    np.save("./pcmci_multi/base_{}_{}_{}_{}_{}.npy".format(f,step,test_start[0],test_end[-1],tau),base_result)
    np.save("./pcmci_multi/model_{}_{}_{}_{}_{}.npy".format(f,step,test_start[0],test_end[-1],tau),model_results)

    np.save("./pcmci_multi/base_p_{}_{}_{}_{}_{}.npy".format(f,step,test_start[0],test_end[-1],tau),base_p)
    np.save("./pcmci_multi/model_p_{}_{}_{}_{}_{}.npy".format(f,step,test_start[0],test_end[-1],tau),model_p)

