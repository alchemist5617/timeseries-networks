library(prophet)
library(forecast)

load("data.RData")
load("lat.RData")
load("lon.RData")


#x<-data[22,8,2:dim(data)[3]]  #cluster==2
#x<-data[28,7,3:dim(data)[3]]   #cluster==0
#x<-data[22,8,]
x<-data[28,7,]
#x<-df_cluster[["X1"]]
#x.ts<-ts(x, start = c(1946, 1), freq=f)

#x <- fuzzify(x)

n=30
# Set forecasting window length (in years)
m=3
# Set start year
start = 1946
origin = 1960
# Set annual sampling rate
f = 12
h = m*f
start_month = 1

index = (origin - start) * f +1

#x.ts <- x[index:length(x)-1]

train_start <- index
train_end <- train_start+n*f-1

test_start <- train_end + 1
test_end <- test_start + m*f -1

x.train <- x[train_start:train_end]
x.test <- x[test_start:test_end]

pa <- phase_average(x.train, f)
x.train.t<- pa$result

A<-prophet.forecst(x.train.t, origin,n,h)
A <- inv_phase_average(A, f, pa$averages, pa$stds)

#A <- prophet.forecst(x.train, origin,n,h)

accuracy(A, x.test)
