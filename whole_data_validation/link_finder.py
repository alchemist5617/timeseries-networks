#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 07:25:29 2020

@author: Mohammad Noorbakhsh
"""

import numpy as np
import feature_finder_f as ff
from datetime import datetime

train_start = np.arange(1926,1977,5)
train_end = np.arange(1955,2006,5)
validation_end = np.arange(1960,2011,5)
test_start = np.arange(1961,2012,5)
test_end = np.arange(1965,2016,5)

n_components_sst = 76


for ijz in range(len(train_start)):
    temporal_limits = {"time_min":datetime(train_start[ijz], 1, 1, 0, 0),"time_max":datetime(validation_end[ijz], 12, 1, 0, 0)}
    count = ff.drought_timeseries("../ET_gamma_18912015.npy",train_start[ijz],validation_end[ijz])
    data_sst, ts, V, df_sst, avg, std = ff.PCA_computer('../sst.mnmean.nc', "sst",temporal_limits, n_components_sst, -9.96921e+36)
    link = ff.PCMCI_generator(ts,count,24)
    
    np.save("./link/link_{}_{}.npy".format(train_start[ijz],validation_end[ijz]), link)
