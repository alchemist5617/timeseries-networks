library(prophet)
library(forecast)
#library(Rssa)
library(ggplot2)
#library(bsts)

load("../data.RData")
load("../lat.RData")
load("../lon.RData")


#x<-data[22,8,2:dim(data)[3]]  #cluster==2
#x<-data[28,7,3:dim(data)[3]]   #cluster==0
#x<-data[22,8,]
x<-data[10,1,]


#bad
x<-data[23,9,]
x<-data[28,7,]

#good
#x<-data[22,19]
#x<-df_cluster[["X1"]]
#x.ts<-ts(x, start = c(1946, 1), freq=f)

#x <- fuzzify(x)

n=30
# Set forecasting window length (in years)
m=3
# Set start year
start = 1946
origin = 1980
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

x.train <- ts(x.train, start = c(origin, 1), freq=f)
x.test <- ts(x.test, start = c(origin+n, 1), freq=f)


stl(x.train,s.window="periodic", robust=TRUE) %>% autoplot()





Precipitation = x.train

d <- data.frame(
  date = seq(as.Date("1983/1/1"), by = "month", length.out = length(Precipitation)),
  Precipitation
)

ggplot(d, aes(x=date)) +                    # basic graphical object
  geom_line(aes(y=Precipitation))+ ylab("Precipitation (mm)")+ xlab("Date")+
  scale_color_discrete(name = "Y series", labels = c("Predictions", "Observations"))+ 
  ggtitle("Monthly Precipitation in equatorial climatic region")+ 
  theme(plot.title = element_text(hjust = 0.5),text = element_text(size=16))

autoplot(x.train)

pa <- phase_average(x.train, f)
x.train.t<- pa$result

A = forecast(auto.arima(x.train.t),h = h)$mean
A <- inv_phase_average(A, f, pa$averages, pa$stds)

accuracy(A, x.test)