library(quantmod)
library(prophet)
library(forecast)

#daily version
prophet.forecst.xts<-function(y, h, colNum = 4, growth_type = 'linear'){
  ds<-index(x) 
  y<-as.numeric(x[,colNum])

  df <- data.frame(ds,y)
  m <- prophet(df, daily.seasonality=TRUE, yearly.seasonality=TRUE, growth = growth_type)
  future <- make_future_dataframe(m, periods = h, freq='month')
  forecast <- predict(m, future)
  plot(m, forecast) + add_changepoints_to_plot(m)
  yhat<-tail(forecast$yhat, n = h)
  return(yhat)
}


test<-aapl["2020-01"]
x.test<-as.numeric(test[,4])
accuracy(y, x.test)

aapl <- getSymbols("AAPL", auto.assign=FALSE)
x<-aapl["2018/2019"]  #data for 2018 and 2019




data <- data.frame(
  date = seq(as.Date("2002/1/1"), by = "month", length.out = 36),
  x.test,
  yhat
)


ggplot(data, aes(x=date)) +                    # basic graphical object
  +     geom_line(aes(y=x.test,colour="red") ) +  # first layer
  +     geom_line(aes(y=yhat,colour="blue"))+ ylab("Values")+ xlab("Date")+
  +     scale_color_discrete(name = "Y series", labels = c("Predictions", "Observations"))
