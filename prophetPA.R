library(prophet)
library(forecast)
library(Rssa)
library(ggplot2)

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



#SSA'

L = 7
s <- ssa(x.train, L = 12)
# Reconstruction stage
# The results are the reconstructed series r$F1, r$F2, and r$F3
recon <- reconstruct(s)
# Calculate the residuals
res <- residuals(recon)


result = matrix(0.0,L,h)
for(i in c(1:L)){
  #lambda <- BoxCox.lambda(mfs[,i])
  #result[i,] = forecast(ets(recon[[i]], biasadj = TRUE),h, biasadj = TRUE)$mean     
  result[i,] =prophet.forecst(recon[[i]], origin,n,h)
}

A = colSums(result)
A[A<0]<-0
#SSA = ts(SSA, start = c(origin+z+n_train, 1), freq=f)

# error.A = accuracy(ABCZT, x.test)
accuracy(A, x.test)



d <- data.frame(
  date = seq(as.Date("1982/1/1"), by = "month", length.out = length(x.test)),
  x.test,
  A
)

ggplot(d, aes(x=date)) +                    # basic graphical object
  geom_line(aes(y=x.test,colour="red") ) +  # first layer
  geom_line(aes(y=A,colour="blue"))+ ylab("Values")+ xlab("Date")+
  scale_color_discrete(name = "Y series", labels = c("Observations", "SSA"))