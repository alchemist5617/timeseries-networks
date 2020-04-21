library(forecast)
library(ncdf4)
library(ggplot2)

fuzzify<-function(x)
{
  idx = (x == 0)
  if(sum(idx) != 0){
    y <- array(0.0 ,length(x))
    y[idx] <- x[idx] + 0.005*runif(1,min=0.0, max=1.0) 
    y[!idx]<-x[!idx] + 0.005*runif(1,min=-1.0, max=1.0)
    return(y)
  }
  return(x) 
}

phase_average<-function(x, freq)
{    
  N = length(x)
  result = numeric(N)
  averages = numeric(freq)
  stds = numeric(freq)    
  for (j in 1:freq){
    Idx = seq(j,N,12)
    averages[j] = mean(x[Idx])
    stds[j] = sd(x[Idx])
    if(stds[j] == 0){
      result[Idx] =  0
    }else{
      result[Idx] = (x[Idx] - averages[j])/stds[j]
    }
    
  }
  returnList <- list("result" = result, "averages" = averages, "stds" = stds)
  return(returnList)  
}

inv_phase_average<-function(x, freq, avg, std)
{    
  N = length(x)
  result = numeric(N)  
  for (j in 1:freq){
    Idx = seq(j,N,freq)
    result[Idx] = x[Idx]* std[j] + avg[j]      
  }
  return(result)  
}

inBoxCox<-function(x, lambda){
  if(lambda == 0){
    inv_x = exp(x)
  } else{
    inx_x = (x*lambda + 1) ^ (1/lambda)
  }
  return(inx_x)
}


load("data.RData")
load("lat.RData")
load("lon.RData")

x<-data[4,5,]

x.ts<-ts(x, start = c(1946, 1), freq=f)

x <- fuzzify(x)

n=20
# Set forecasting window length (in years)
m=3
# Set start year
start = 1946
origin = 1982
# Set annual sampling rate
f = 12
h = m*f

index = (origin - (start - 1)) * 12 +1

x.ts <- x[index:length(x)-1]

train_start <- index
train_end <- train_start+n*f-1

test_start <- train_end + 1
test_end <- test_start + m*f -1

x.train <- x[train_start:train_end]
x.test <- x[test_start:test_end]

x.ts = ts(x, start=c(origin, 1), freq=f)

x.train <- ts(x.train, start = c(origin, 1), freq=f)
x.test <- ts(x.test, start = c(origin+n, 1), freq=f)
