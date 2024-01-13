

# loading the libraries 
library(ggplot2)
library(ggthemes)
library(dplyr)
library(caTools)
library(corrplot)


df = read.table("startups.txt",sep = ',',header = TRUE)
# head(df)

# splitting the data set
sample<-sample.split(df$Profit,SplitRatio = 0.7)
train=subset(df,sample==TRUE)
test=subset(df,sample==FALSE)

# getting the structure of train data 
str(train)

#checking the null values 
any(is.na(train))

cols.numeric.train = sapply(train, is.numeric)
cols.numeric.train



# getting the correlation of the numeric features 
correlation.train = cor(train[,cols.numeric.train])
correlation.train


# plotting the heat map of correlation 
corrplot(correlation.train,method = 'color')


# creating the first model 
lm_model1 = lm(Profit~  R.D.Spend+Administration+Marketing.Spend,data = train)
# getting the summary of the first model
summary(lm_model1)


# creating another model 
lm_model2 = lm(Profit~R.D.Spend+Marketing.Spend,data = train)
# getting the summary of the first model
summary(lm_model2)


# saving the model residuals
res = residuals(lm_model1)
res = as.data.frame(res)
head(res)


# plotting the residual
pl_residuals = ggplot(res,aes(res))+geom_histogram()+ theme_minimal()
pl_residuals


# testing the model in data set
profit.prediction = predict(lm_model1,test)
profit.prediction = as.data.frame(profit.prediction)
head(profit.prediction)


# saving predicted and actual in a new data frame 
result = cbind(profit.prediction, test$Profit)
colnames(result) = c('pred', 'real')
result = as.data.frame(result)
result

# plotting residuals of test data  
pl_residuals_test = ggplot(profit.prediction,aes(profit.prediction))+geom_histogram()
+ theme_minimal()
pl_residuals_test


# getting the mean square value
mse = mean((result$real - result$pred)^2)
print(mse)


# calculating the rmse
rmse = mse^0.5
print(rmse)

# finding the sse and tss , r^2 
sse = sum((result$pred - result$real)^2)
sst = sum((mean(test$Profit) - result$real)^2)
r2 = 1 - sse/sst
print(r2)

