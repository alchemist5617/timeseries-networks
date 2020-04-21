prophet.forecst1<-function(y,year,month, n, h){
  
  start_date = paste(year,month,"01", sep = "-")
  print(start_date)
  if(month==1){
    month=12
    end_year = year+n-1
  }else{
    month = month -1
    end_year = year+n
  }
  end_date = paste(end_year,month,"01", sep = "-")
  print(end_date)
  ds = seq(from = as.Date(start_date), to = as.Date(end_date), by = 'month')
  df <- data.frame(ds,y)
  
  m <- prophet(df, weekly.seasonality=FALSE, daily.seasonality = FALSE)
  future <- make_future_dataframe(m, periods = h, freq='month')
  forecast <- predict(m, future)
  yhat<-tail(forecast$yhat, n = h)
  return(yhat)
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

load("data.RData")
load("lat.RData")
load("lon.RData")

n_train = 30
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

for(i in c(1:1)){
  for(j in c(10:10)){
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
        
        #APA = forecast(auto.arima(x.train.t),h=h)$mean
        #APA = inv_phase_average(APA, f, pa$averages, pa$stds)
        
        #EPA = forecast(ets(x.train.t),h=h)$mean
        #EPA = inv_phase_average(EPA, f, pa$averages, pa$stds)
        
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
        
        #month <- ((z-1) %% f) + 1
        #year <- ((z-1) %/% f) + start_year
        
        
        pa <- phase_average(x.train, f)
        x.train.t<- pa$result
        
        APA<-prophet.forecst(x.train.t,start_year + z-1,n_train,h)
        APA <- inv_phase_average(APA, f, pa$averages, pa$stds)
        
        
        # error.A = accuracy(ABCZT, x.test)
        error.E = accuracy(APA, x.test)
        
        # RMSE.A[z] = as.numeric(error.A[1,2])
        # MAE.A[z] = as.numeric(error.A[1,3])
        # errorRatio.A[z] = RMSE.A[z]/mean(x.test)
        
        
        RMSE.E[z] = as.numeric(error.E[1,2])
        MAE.E[z] = as.numeric(error.E[1,3])
        errorRatio.E[z] = RMSE.E[z]/mean(x.test)
      }
      # Now calculate average forecast errors
      #meanRMSE.A = mean(RMSE.A, na.rm=TRUE)
      meanRMSE.E = mean(RMSE.E, na.rm=TRUE)
      # meanErrorRatio.A = mean(errorRatio.A)
      
      # meanMAE.A = mean(MAE.A, na.rm=TRUE)
      meanMAE.E = mean(MAE.E, na.rm=TRUE)
      meanErrorRatio.E = mean(errorRatio.E, na.rm=TRUE)
      
      # de.A<-data.frame(i,j,lon[i],lat[j],meanRMSE.A, meanMAE.A, meanErrorRatio.A)
      de.E<-data.frame(i,j,lon[i],lat[j],meanRMSE.E, meanMAE.E, meanErrorRatio.E)
      
      # names(de.A)<-c("i","j","lon","lat","meanRME", "meanMAE","meanErrorRatio")
      names(de.E)<-c("i","j","lon","lat","meanRME", "meanMAE","meanErrorRatio")
      
      # write.table(de.A, "ABCZTTemp.csv", sep = ",", col.names = !file.exists("ABCZTTemp.csv"), append = T)
      write.table(de.E, "./AEEMDTemp30.csv", sep = ",", col.names = !file.exists("./AEEMDTemp30.csv"), append = T)
      
      # df.A <- rbind(df.A,de.A)
      df.E <- rbind(df.E,de.E)
    }
  }m
}
