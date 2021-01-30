import numpy as np
import feature_finder as ff
from datetime import datetime

train_start = np.arange(1926,1977,5)
train_end = np.arange(1955,2006,5)
validation_end = np.arange(1960,2011,5)
test_start = np.arange(1961,2012,5)
test_end = np.arange(1965,2016,5)

n_components_sst = 76
Index = 360
taus = np.arange(-1,-13,-1)
base_result = []
model_results= []

final_base = []
final_model = []

for tau in taus:
    for ijz in range(len(train_start)):
        temporal_limits = {"time_min":datetime(train_start[ijz], 1, 1, 0, 0),"time_max":datetime(train_end[ijz], 12, 1, 0, 0)}
        count = ff.drought_timeseries("ET_gamma_18912015.npy",train_start[ijz],train_end[ijz])
        data_sst, ts, V, df_sst, avg, std = ff.PCA_computer('../sst.mnmean.nc', "sst",temporal_limits, n_components_sst, -9.96921e+36)
        link = ff.PCMCI_generator(ts,count,12)
        
#        temporal_limits = {"time_min":datetime(train_start[ijz], 1, 1, 0, 0),"time_max":datetime(validation_end[ijz], 12, 1, 0, 0)}
 #       data_sst = ff.data_generator('../sst.mnmean.nc', "sst",temporal_limits, -9.96921e+36)
 #       count = ff.drought_timeseries("ET_gamma_18912015.npy",train_start[ijz],validation_end[ijz])
        best_link, base_model, model = ff.best_link_finder(count, data_sst, link, df_sst, V, index = Index , tau=tau)

        temporal_limits = {"time_min":datetime(test_start[ijz], 1, 1, 0, 0),"time_max":datetime(test_end[ijz], 12, 1, 0, 0)}
        data_sst = ff.data_generator_avg_std('../sst.mnmean.nc', "sst",temporal_limits, avg, std, 12, -9.96921e+36)
        count = ff.drought_timeseries("ET_gamma_18912015.npy",test_start[ijz],test_end[ijz])
        
        base_result.append(ff.base_model_result(count, base_model, tau))
        model_results.append(ff.model_result(count, data_sst, best_link, df_sst, V, model,tau))
    
    np.save("base_result_V_{}.npy".format(np.abs(tau)),base_result)
    np.save("model_results_V_{}.npy".format(np.abs(tau)),model_results)
    
    final_base.append(np.nanmean(base_result))
    final_model.append(np.nanmean(model_results))
    
np.save("results.npy",np.array(list(zip(final_base,final_model))))