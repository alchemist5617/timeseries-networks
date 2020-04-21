library(forecast)



# Functions
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

#data <- read.csv("precipitation.csv", header=TRUE, sep=",")
#lon <- read.csv("lon.csv", header=TRUE, sep=",")
#lat <- read.csv("lat.csv", header=TRUE, sep=",")

#lon<-lon[["X0"]]
#lat<-lat[["X0"]]

load("data.RData")
load("lat.RData")
load("lon.RData")



n_train = 20
# Set forecasting window length (in years)
n_test = 3
# Set start year
start_year = 1946
# Set annual sampling rate
f = 12
h = n_test * f 

skill <- list()

length.lon = dim(data)[1]
length.lat = dim(data)[2]
L = dim(data)[3]

end_year = start_year + ((L -(L %% f))/f)

df.A<-data.frame(as.numeric,as.numeric,as.numeric, as.numeric, as.numeric, as.numeric, as.numeric )
names(df.A)<-c("i","j","lon","lat","RMSE","MAE","meanErrorRatio")

df.E<-data.frame(as.numeric,as.numeric,as.numeric, as.numeric, as.numeric, as.numeric, as.numeric)
names(df.E)<-c("i","j","lon","lat","RMSE","MAE","meanErrorRatio")

#for(j in c(2:L)){
  
for(i in c(1:length.lon)){
  for(j in c(1:length.lat)){
    if(!is.na(data[i,j,L])){
      
      x <- data[i,j,]
      x <- sapply(x, fuzzify)
      
      RMSE.A = array(0.0, (end_year-(n_train+n_test))-start_year+1)
      MAE.A = array(0.0, (end_year-(n_train+n_test))-start_year+1)
      errorRatio.A = array(0.0, (end_year-(n_train+n_test))-start_year+1)
      
      RMSE.E = array(0.0, (end_year-(n_train+n_test))-start_year+1)
      MAE.E = array(0.0, (end_year-(n_train+n_test))-start_year+1)
      errorRatio.E = array(0.0, (end_year-(n_train+n_test))-start_year+1)
      
      x.ts <- x[1:length(x)-1]
      origin = start_year - 1
      N = length(x.ts)
      
      for(z in seq(1,N/f - n_train - n_test + 1)){
        
        train_start = (z - 1)*f + 1
        train_end = train_start + n_train*f - 1
        
        test_start = train_end + 1
        test_end = test_start + n_test*f - 1
        
        
        x.train <- x[train_start:train_end]
        x.test <- x[test_start:test_end]
        
        x.train <- ts(x.train, start = c(origin+z, 1), freq=f)
        x.test <- ts(x.test, start = c(origin+z+n_train, 1), freq=f)
        
        lambda <- BoxCox.lambda(x.train)
        #OM
        #A = forecast(auto.arima(x.train),h=h)$mean
        #E = forecast(ets(x.train),h=h)$mean
        
        #OM + BC
        #ABC = forecast(auto.arima(x.train,lambda=lambda), h=h, lambda=lambda, biasadj=TRUE)$mean
        #EBC = forecast(ets(x.train,lambda=lambda), h=h, lambda=lambda, biasadj=FALSE)$mean
        
        #OM + BC + ZT
     #   x.train.t <- BoxCox(x.train, lambda)
    #    m = mean(x.train.t)
    #    s = sd(x.train.t)
    #    x.train.t = (x.train.t - m)/s
        
    #    ABCZT = forecast(auto.arima(x.train.t),h=h)$mean
    #    ABCZT = ABCZT*s + m
    #    ABCZT = invBoxCox(ABCZT,lambda)
        
    #    EBCZT = forecast(ets(x.train.t),h=h)$mean
    #    EBCZT = EBCZT*s + m
    #    EBCZT = invBoxCox(EBCZT,lambda)
        
    
    #OM + PA
    pa = phase_average(x.train, f)
    x.train.t <- ts(pa$result, start = c(origin, 1), freq=f)
    
    APA = forecast(auto.arima(x.train.t),h=h)$mean
    APA = inv_phase_average(APA, f, pa$averages, pa$stds)
    
    EPA = forecast(ets(x.train.t),h=h)$mean
    EPA = inv_phase_average(EPA, f, pa$averages, pa$stds)
    
    #OM + BC + PA
    #x.train.t <- BoxCox(x.train, lambda)
    #pa = phase_average(x.train.t, f)
    #x.train.t <- ts(pa$result, start = c(origin, 1), freq=f)
    
    #ABCPA = forecast(auto.arima(x.train.t),h=h)$mean
    #ABCPA = inv_phase_average(ABCPA, f, pa$averages, pa$stds)
    #ABCPA = invBoxCox(ABCPA,lambda)
    
    #EBCPA= forecast(ets(x.train.t),h=h)$mean
    #EBCPA = inv_phase_average(EBCPA, f, pa$averages, pa$stds)
    #EBCPA = invBoxCox(EBCPA,lambda)
    
    #E + STL
    #ES = forecast(stl(x.train,s.window="periodic", robust = TRUE), h=h, biasadj=FALSE)$mean
    
    #E + BC + STL
    #x.fit = stl(BoxCox(x.train,lambda), s.window="periodic", robust = TRUE)
    #EBCSTL = forecast(x.fit, h=h, lambda=lambda, biasadj=FALSE)$mean

    error.A = accuracy(APA, x.test)
    error.E = accuracy(EPA, x.test)
    
    RMSE.A[z] = as.numeric(error.A[1,2])
    MAE.A[z] = as.numeric(error.A[1,3])
    errorRatio.A[z] = RMSE.A[z]/mean(x.test)


    RMSE.E[z] = as.numeric(error.E[1,2])
    MAE.E[z] = as.numeric(error.E[1,3])
    errorRatio.E[z] = RMSE.E[z]/mean(x.test)

  }
  
  # Now calculate average forecast errors
  meanRMSE.A = mean(RMSE.A, na.rm=TRUE)
  meanRMSE.E = mean(RMSE.E, na.rm=TRUE)
  meanErrorRatio.A = mean(errorRatio.A)
  
  meanMAE.A = mean(MAE.A, na.rm=TRUE)
  meanMAE.E = mean(MAE.E, na.rm=TRUE)
  meanErrorRatio.E = mean(errorRatio.E)
  
  de.A<-data.frame(i,j,lon[i],lat[j],meanRMSE.A, meanMAE.A, meanErrorRatio.A)
  de.E<-data.frame(i,j,lon[i],lat[j],meanRMSE.E, meanMAE.E, meanErrorRatio.E)
  
  names(de.A)<-c("i","j","lon","lat","meanRME", "meanMAE","meanErrorRatio")
  names(de.E)<-c("i","j","lon","lat","meanRME", "meanMAE","meanErrorRatio")
  
  write.table(de.A, "APATemp.csv", sep = ",", col.names = !file.exists("ABCZT.csv"), append = T)
  write.table(de.E, "EPATemp.csv", sep = ",", col.names = !file.exists("EBCZT.csv"), append = T)
  
  df.A <- rbind(df.A,de.A)
  df.E <- rbind(df.E,de.E)
    }
  }
}
write.csv(df.A, file = "APA.csv")
write.csv(df.E, file = "EPA.csv")
