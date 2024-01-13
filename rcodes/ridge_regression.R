

# loading the libraries 
library(ggplot2)
library(glmnet)

# setting the seed 
set.seed(42)

# loading the data frame 
data = read.table('nse.csv',sep = ',',header = TRUE)

# attaching the data 
attach(data)

df = data[,c('Open', 'High', 'Low','Close', 'Total.Trade.Quantity','Turnover_Lacs')]

# converting the data to numerical matrix
dm = data.matrix(df, rownames.force = NA)

# dividing the trade column by 10,000
dm[,'Total.Trade.Quantity'] = dm[,'Total.Trade.Quantity']/100000

# converting back to dataframe
dm = as.data.frame(dm)

## splitting the data into train and test
library(caTools)
sample = sample.split(dm$Open,SplitRatio = 0.7)
train = subset(dm, sample ==TRUE)
test = subset(dm, sample == FALSE)


# getting the correlation
# print(cor(dm))


# creating the first model
model1 = lm(Turnover_Lacs~.,data = train )
summary(model1)


# checking the residuals of our model
res = residuals(model1)
res = as.data.frame(res)

# plotting the residuals
ggplot(res,aes(res))+geom_histogram(fill='dark green',alpha=0.5)


turn_predictions = predict(model1, test)
result = cbind(turn_predictions, test$Turnover_Lacs)
colnames(result) = c('pred', 'real')
result = as.data.frame(result)



# calculating the mean square error
mse = mean((result$real - result$pred)^2)
print(mse)


# calculating the rmse
rmse = mse^0.5
print(rmse)


# calculating the sse and sst and then calcualting the coefficient of regression
sse = sum((result$pred - result$real)^2)
sst = sum((mean(train$Turnover_Lacs )- result$real)^2)
r2 = 1 - (sse/sst)
print(r2)



# ridge regression for overcoming overfitting
library(dplyr)
library(caret)

num.cols = sapply(dm,is.numeric)
dummies = dummyVars(Turnover_Lacs ~., data = dm[,num.cols])
print(dummies)


train_dummies = predict(dummies, newdata = train[,num.cols])
test_dummies = predict(dummies,newdata = test[,num.cols])
print(dim(train_dummies))


print(dim(test_dummies))


x = as.matrix(train_dummies)
x_test = as.matrix(test_dummies)

y_train = train$Turnover_Lacs
y_test = test$Turnover_Lacs
lambdas = 10^seq(2,-3,by = -.1)
ridge_reg = glmnet(x,y_train, nlambda = 25, alpha = 0, family= 'gaussian', lambda =
                     lambdas)
summary(ridge_reg)


# getting the optimal value of lambda
cv_ridge <- cv.glmnet(x, y_train, alpha = 0, lambda = lambdas)
optimal_lambda <- cv_ridge$lambda.min
optimal_lambda


# Compute R^2 from true and predicted values
eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  # Model performance metrics
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
}

# Prediction and evaluation on train data
predictions_train <- predict(ridge_reg, s = optimal_lambda, newx = x)
eval_results(y_train, predictions_train, train)


# Prediction and evaluation on test data
predictions_test <- predict(ridge_reg, s = optimal_lambda, newx = x_test)
eval_results(y_test, predictions_test, test)







