
library(datarium)

# loading the data 
df = marketing

# creating the linear model 
model = lm(sales ~ youtube, data = df)
model

# anova and f-stat calculation  : manual 
attach(df)

# residuals / rss/ ess/ tss / f_stat ( manually )
tss = sum((sales - mean(sales))^2) 
res = residuals(model)
rss = sum(res^2)
ess = tss -rss
f_stat = (ess/1) / (rss/(200-2))


# summary of the model 
summary(model)


# generating predicted values and checking residual plot 
preds = predict(model)
plot(preds, res)

# creating the multiple regression model(model_mlr)
library(datarium)
df = marketing
model_mlr = lm(sales ~. , data = df)
summary(model_mlr)


# formal testing of heteroskedasticity (breusch pagan test - imtest package)
library(lmtest)
lmtest::bptest(model)

library(skedastic)
skedastic::breusch_pagan(model)


# white test 
skedastic::white_lm(model)




