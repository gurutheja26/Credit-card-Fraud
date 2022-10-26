#Importing Dataset
data <- read.csv("C:/Users/HP/Desktop/creditcard.csv")
print(head(data))


#Data Pre-processing
print(class(data)) # class of data object
print(str(data)) # compact display of the internal structure of data
print(summary(data)) # summary of data
print(names(data)) # column names
print(dim(data)) # dimensions of data
print(head(data)) # view of the data from the top
print(tail(data)) # view of the data from the bottom


#Data Cleaing
library('dplyr')
library("stringr")
data%>% mutate_if(is.character, str_trim) -> data
df=data.frame(data)
print(head(df))
print(is.na(df))
print(any(is.na(df)))
print(sum(is.na(df)))
print(complete.cases(df))
print(df[complete.cases(df),])
df=na.omit(df)
print(df)


#Importing Required Packages
install.packages("ranger")
install.packages("caret")
install.packages("data.table")
library("ranger")
library("caret")
library("data.table")
#Importing Dataset
data <- read.csv("C:/Users/HP/Desktop/creditcard.csv")
df<-data.frame(data)


#dataExploration
print(class(df)) # class of data object
print(str(df)) # compact display of the internal structure of data
print(summary(df)) # summary of data
print(names(df)) # column names
print(dim(df)) # dimensions of data
print(head(df)) # view of the data from the top
print(tail(df)) # view of the data from the bottom
print(table(df$Class))
print(var(df$Amount))
#Mean of amount column
mean(df$Amount)
#Standard Deviation of Amount column
sd(df$Amount)
#Variance of Amount column
#Median of Amount Column
median(df$Amount)

#Data Manipulation
head(df)
tail(df)

df$Amount=scale(df$Amount)
data1=df[,-c(1)]
print(head(data1))

#DataModelling
install.packages("caTools")
library("caTools")
set.seed(123)
data2 = sample.split(data1$Class,SplitRatio=0.80)
traindata = subset(data1,data2==TRUE)
testdata = subset(data1,data2==FALSE)
dim(traindata)
dim(testdata)
#Scatterplot between v1 and amount
ggplot(testdata, aes(x = Amount, y =V1)) + 
  geom_point()
#Scatter plot with color between group V1 and Amount and group by Class
ggplot(traindata, aes(x = V1, y =V2)) + geom_point(aes(color = factor(Class)))


# BOXPLOT ON TESTDATA
b1<- ggplot(testdata, aes(x = Class, y = V1))
# Add the geometric object box plot
b1 + geom_boxplot()
#Histogram Between Class and Column V1 on testdata
ggplot(testdata, aes(x = Class, y = V1)) + geom_bar(stat = "identity") 

#BarPlot Of Column Amount
ggplot(traindata[1:100,c(25:30)], aes(x = factor(Amount))) +
  geom_bar(fill = "orange") + theme_classic()

#Logistic Regression
# Loading package
library(dplyr)
# Loading package
library(caTools)
install.packages("ROCR")
library("ROCR") 
# Training model
logistic_model <- glm(Class~., 
                      data = traindata, 
                      family = "binomial")
logistic_model
# Summary

summary(logistic_model)

#Plot
plot(logistic_model)
# Predict test data based on model
predict_reg <- predict(logistic_model, 
                       testdata, type = "response")
predict_reg  

# Changing probabilities
predict_reg <- ifelse(predict_reg >0.5, 1, 0)
# Evaluating model accuracy
# using confusion matrix
table(testdata$Class, predict_reg)

missing_classerr <- mean(predict_reg != testdata$Class)
print(paste('Accuracy =', 1 - missing_classerr))

# ROC-AUC Curve
ROCPred <- prediction(predict_reg, testdata$Class) 
ROCPer <- performance(ROCPred, measure = "tpr", 
                      x.measure = "fpr")

auc <- performance(ROCPred, measure = "auc")
auc <- auc@y.values[[1]]
auc

# Plotting curve
plot(ROCPer)

plot(ROCPer, colorize = TRUE, 
     print.cutoffs.at = seq(0.1, by = 0.1), 
     main = "ROC CURVE")

abline(a = 0, b = 1)
auc <- round(auc, 4)
legend(.6, .4, auc, title = "AUC", cex = 1)


#Decision Tree
install.packages("rpart.plot")
library("rpart.plot")
library(rpart)
fit <- rpart(Class~., data = traindata, method = 'class')
rpart.plot(fit, extra = 106)

predict_model<-predict(fit, testdata, type = 'class')
table_mat <- table(testdata$Class, predict_model)
table_mat

accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
print(paste('Accuracy for test', accuracy_Test))

predict_model<-predict(fit, traindata, type = 'class')
table_mat <- table(traindata$Class, predict_model)
table_mat

accuracy_Train<- sum(diag(table_mat)) / sum(table_mat)
print(paste('Accuracy for train data', accuracy_Train))

#ArtificialNeuralNetwork
install.packages("neuralnet")
library("neuralnet")
ann_mod=neuralnet (Class~.,traindata,linear.output=FALSE)


# error
ann_mod$result.matrix







# Prediction


head(traindata[1, ])
# confusion Matrix $Misclassification error -Training data
output <- compute(ann_mod, rep = 1, traindata[, -1])
p1 <- output$net.result
pred1 <- ifelse(p1 > 0.5, 1, 0)
tab1 <- table(pred1, traindata$Class)
tab1

#mis-classification error
1 - sum(diag(tab1)) / sum(tab1)

#GradientBoosting
library(gbm, quietly=TRUE)

# Get the time to train the GBM model
system.time(
  model_gbm <- gbm(Class ~ .
                   , distribution = "bernoulli"
                   , data = rbind(traindata, testdata)
                   , n.trees = 500
                   , interaction.depth = 3
                   , n.minobsinnode = 100
                   , shrinkage = 0.01
                   , bag.fraction = 0.5
                   , train.fraction = nrow(traindata) / (nrow(traindata) + nrow(testdata))
  )
)
# Determine best iteration based on test data
gbm.iter = gbm.perf(model_gbm, method = "test")
model.influence = relative.influence(model_gbm, n.trees = gbm.iter, sort. = TRUE)
#Plot the gbm model
plot(model_gbm)

# Plot and calculate AUC on test data
gbm_test = predict(model_gbm, newdata = testdata, n.trees = gbm.iter)
gbm_auc = roc(testdata$Class, gbm_test)
plot(gbm_auc,main=paste0("AUC: ",round(pROC::auc(gbm_auc),3)), col = "red")
print(gbm_auc)
model_gbm


























  









