

# loading the dataset 
df.train <- read.csv('/home/thomaskutty/Gitrepo/Machine-Learning-with-R/data_fold/admission.csv')
df.train <- subset(df.train, select = -c(Serial.No.) )
# print(head(df.train))


# attaching the dataset 
attach(df.train)

# creating a function to classify the last chance of admit column into 1 and 0 
to_class = function(x){
  if (x>0.5){
    1
  }else{
    0
  }
}


# adding a new column to our training data set ( giving the class labels)
df.train$class <- sapply(df.train$Chance.of.Admit, to_class)
# print(head(df.train))


library(GGally)
# printing the correlation 
ggcorr(df.train, label = T, hjust = 0.6, cex=3 ,color = 'blue')


#lets make the class column factor 
df.train$class = factor(df.train$class)
# removing the chance of admit column ( already encoded this column into categorical classes)
df.train = subset(df.train, select = c(-Chance.of.Admit))
# head(df.train)

# visualization section using ggplot 
library(ggplot2)
# plot
cgpa_plt = ggplot(df.train, aes(CGPA))+ geom_histogram(fill = '#696aaa' , bins = 20) + theme_minimal()

gre_toefl_plt = ggplot(df.train, aes(x = GRE.Score, y = TOEFL.Score)) + geom_point()+ theme_minimal()

research_class_plt = ggplot(df.train,aes(Research))+geom_bar(aes(fill=factor(class)),alpha=0.5) + theme_minimal()

university_ranks_class_plt = ggplot(df.train,aes(University.Rating))+geom_bar(aes(fill=factor(class)),alpha=0.5) + theme_minimal()




library(caTools)
set.seed(42)
# splitting the data set into train and test 
split = sample.split(df.train$class, SplitRatio  = 0.70)
final.train = subset(df.train, split = TRUE)
final.test = subset(df.train, split = FALSE)


# creating the logistic regression model 
final.log.model = glm(formula = class~., family = binomial(link = 'logit'), data = final.train)
summary(final.log.model)

# predicting the values for the test data 
fitted.probabilities<-predict(final.log.model,newdata = final.test,type='response')
# converting it into train and test 
fitted.results<-ifelse(fitted.probabilities>0.5,1,0)

# getting the accuracy 
misclassicerror<-mean(fitted.results!=final.test$class,na.rm=T)
#misclassicerror
print(paste('accuracy',1-misclassicerror))

# creating a confusion matrix ( consisting true positives, true negatives, false positives, false negatives)
table(final.test$class,fitted.probabilities>0.5)


library("pROC")
# storing the probabilities for the test data into a new vairable 
test_prob = predict(final.log.model, newdata = final.test, type = "response")
# using the test_probabilites and original classes , computing the roc and auc 
test_roc = roc(final.test$class ~ test_prob, plot = TRUE, print.auc = TRUE)


### Interpretation: 
#=======================================================================================
# We can see that our AUC (AREA UNDER CURVE ) if 0.9423 where 1.0 represents perfect classifier and 0.5
# represents worth less classifier .
# AUC is 0.9423 means that there is 94.23% chance that model will be able to distinguish between positive class and
# negative class.




