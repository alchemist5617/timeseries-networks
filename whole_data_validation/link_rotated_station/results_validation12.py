import numpy as np
import feature_finder_f as ff
from datetime import datetime
from sklearn.metrics import mean_squared_error

train_start = np.arange(1926,1977,5)
train_end = np.arange(1955,2006,5)
validation_end = np.arange(1960,2011,5)
test_start = np.arange(1961,2012,5)
test_end = np.arange(1965,2016,5)

n_components_sst = 98
Index = 360
taus = np.arange(1,13,1)


for tau in taus:
#    final_base = []
#    final_model = []
    base_result = []
    model_results= []

    
    for ijz in range(len(train_start)):
        temporal_limits = {"time_min":datetime(train_start[ijz], 1, 1, 0, 0),"time_max":datetime(validation_end[ijz], 12, 1, 0, 0)}
        original_count, count = ff.drought_timeseries("../ET_gamma_18912015.npy",train_start[ijz],validation_end[ijz])
        data_sst, ts, V, df_sst, avg, std = ff.PCA_computer_rotated('../sst.mnmean.nc', "sst",temporal_limits, n_components_sst, -9.96921e+36)
        link = np.load("./link_rotated/link_{}_{}_{}.npy".format(train_start[ijz],validation_end[ijz],np.abs(tau)))
        
#        with open("./new/list_V_{}.txt".format(np.abs(tau)), "a") as myfile:
#            for p in range(len(link)):
#                myfile.write(str(link[p,:])+"\n")
#            myfile.write("***\n")
#        temporal_limits = {"time_min":datetime(train_start[ijz], 1, 1, 0, 0),"time_max":datetime(validation_end[ijz], 12, 1, 0, 0)}
 #       data_sst = ff.data_generator('../sst.mnmean.nc', "sst",temporal_limits, -9.96921e+36)
 #       count = ff.drought_timeseries("ET_gamma_18912015.npy",train_start[ijz],validation_end[ijz])
        best_link, base_model, model = ff.forward_feature_V(original_count, data_sst, link, V, tau=tau, ratio=0.8)

        temporal_limits = {"time_min":datetime(test_start[ijz], 1, 1, 0, 0),"time_max":datetime(test_end[ijz], 12, 1, 0, 0)}
        data_sst = ff.data_generator_avg_std('../sst.mnmean.nc', "sst",temporal_limits, avg, std, 12, -9.96921e+36)
        original_count, count = ff.drought_timeseries("../ET_gamma_18912015.npy",test_start[ijz],test_end[ijz])
        
        y_pred_base, y_test_base = ff.base_model_result(original_count, base_model, best_link, tau)
        y_pred, y_test = ff.model_result(original_count, data_sst, best_link, df_sst, V, model,tau)
        
        np.save("./new_rotated/pred_base_{}_{}_{}.npy".format(test_start[ijz],test_end[ijz],tau),y_pred_base)
        np.save("./new_rotated/test_base_{}_{}_{}.npy".format(test_start[ijz],test_end[ijz],tau),y_test_base) 

        np.save("./new_rotated/pred_{}_{}_{}.npy".format(test_start[ijz],test_end[ijz],tau),y_pred) 
        np.save("./new_rotated/test_{}_{}_{}.npy".format(test_start[ijz],test_end[ijz],tau),y_test)


        base_result.append(mean_squared_error(y_pred_base, y_test_base))
        model_results.append(mean_squared_error(y_pred, y_test))
    
    np.save("./new_rotated/base_V_{}.npy".format(np.abs(tau)),base_result)
    np.save("./new_rotated/model_V_{}.npy".format(np.abs(tau)),model_results)
    
 #   final_base.append(np.nanmean(base_result))
 #   final_model.append(np.nanmean(model_results))
    
#np.save("./new/results_V.npy",np.array(list(zip(final_base,final_model))))
