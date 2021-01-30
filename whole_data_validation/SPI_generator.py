#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 08:57:17 2020

@author: Mohammad Noorbakhsh
"""
import numpy as np
import scipy.stats as st
import pandas as pd

class SPI:
    
    def __init__(self, rolling_n = 12, f = 12, n = 30):
        self.rolling_n = rolling_n
        self.f = f
        self.n = n
        
    def _spiParametersMle(x, dist=st.gamma):
        i = x == 0
        params = dist.fit(x[~i])
        q = len(x[i])/len(x)
        return(params, q)
    
    def _spiGeneratorMle(x, dist=st.gamma):
        i = x == 0
    
        params = dist.fit(x[~i])
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        G = dist.cdf(x[~i], loc=loc, scale=scale, *arg)
    
        q = len(x[i])/len(x)
        probabilities = np.zeros(len(x))
        probabilities[i] = q
        probabilities[~i] = q + (1 - q) * G
        result = st.norm.ppf(probabilities)
        return(result)
    
    def _spiGeneratorParamMle(x, params, q, dist=st.gamma):
        i = x == 0
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        G = dist.cdf(x[~i], loc=loc, scale=scale, *arg)
    
        if G == 1: G = 0.99
        probabilities = np.zeros(len(x))
        probabilities[i] = q
        probabilities[~i] = q + (1 - q) * G
        result = st.norm.ppf(probabilities)
        return(result)

    def spi_generator(self, file_name):
        result = np.load(file_name)
        result = pd.DataFrame(result)
        
        RFThree = result.rolling(self.rolling_n).apply(sum)
        RFThree = RFThree.iloc[self.rolling_n - 1:,:]
        
        N = RFThree.shape[0]       
        d3 = N - (self.n*self.f + 1)
        
        result_index = []
        for k in range(d3):
            onset = k
            end = k + (self.n*self.f - (self.rolling_n - 1))
        
            a = RFThree.iloc[onset:end,:].values
            b = RFThree.iloc[end + (self.rolling_n - 1),:].values
            n_a = a.shape[1]
            index = np.zeros(n_a)
        
            for i in range(n_a):
                x = a[:,i]
                params,q = self.spiParametersMle(x)
                r = self.spiGeneratorParamMle([b[i]], params, q)
                index[i] = r
        
            result_index.append(index)
            return(np.array(result_index))

