library(prophet)
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

prophet.forecst<-function(y, start_year, n, h){
 start_date = paste(start_year,"01","01", sep = "-") 
 end_date = paste(start_year+n-1,"12","01", sep = "-")

 ds = seq(from = as.Date(start_date), to = as.Date(end_date), by = 'month')
 df <- data.frame(ds,y)
 m <- prophet(df, weekly.seasonality=FALSE, daily.seasonality = FALSE)
 future <- make_future_dataframe(m, periods = h, freq='month')
 forecast <- predict(m, future)
 yhat<-tail(forecast$yhat, n = h)
 return(yhat)
}


load("../data.RData")
load("../lat.RData")
load("../lon.RData")

x<-data[28,7,]

n=20
# Set forecasting window length (in years)
m=3
# Set start year
start = 1977
origin = 1980
# Set annual sampling rate
f = 12
h = m*f

index = (origin - start) * f +1

#x.ts <- x[index:length(x)-1]

train_start <- index
train_end <- train_start+n*f-1

test_start <- train_end + 1
test_end <- test_start + m*f -1

x.train <- x[train_start:train_end]
x.test <- x[test_start:test_end]

ds = seq(from = as.Date("1977-01-01"), to = as.Date("1996-12-01"), by = 'month')

pa <- phase_average(x.train, f)
y<- pa$result
y<-x.train

df <- data.frame(ds,y)

m <- prophet(df)
future <- make_future_dataframe(m, periods = 36, freq='month')

forecast <- predict(m, future)

#plot(m, forecast)

#prophet_plot_components(m, forecast)

yhat<-tail(forecast$yhat, n = h)

yhat = inv_phase_average(yhat, f, pa$averages, pa$stds)

accuracy(yhat,x.test)

