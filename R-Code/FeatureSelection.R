# Feature Selection with Caret R Package
#######################################################
# 1. Remove Redundant Feature
#######################################################
# ensure results are repeatable
set.seed(101)
# load library 
#install.packages("mlbench")
#install.packages("caret")
library(mlbench)
library(caret)

# load data
data_2018 = read.csv(file = '/Users/manishparihar/Desktop/Infura/ethereum2018.csv', header = TRUE)
data_2018

# drop column
drop <- c("PrivateNote","ContractAddress","Status","ErrCode")
df = data_2018[,!(names(data_2018) %in% drop)]
df

# drop unnecessary non numeric data
removeTranX_datadrop <- c("Txhash","DateTime","From","To")
dropNonNumericData = df[,!(names(df) %in% removeTranX_datadrop)]
# final dataset
dropNonNumericData

# laod data
data(dropNonNumericData)
# calculate correlation matrix
correlationMatrix <- cor(dropNonNumericData)
# summarize correlation matrix
print(correlationMatrix)
# find attributes that are highly correacted (ideally > 0.75)
higherCorelated <- findCorrelation(correlationMatrix, cutoff = 0.5)
# print indexes of highly correlated attributes
print(higherCorelated)
# Output is -  1 2 8 6 5 

#######################################################
# 2. Rank Features By Importance
#######################################################
set.seed(102)
# load the dataset
data(dropNonNumericData)
# training scheme preparation
# cross validation : 10 folds provide commonly used 
# tradeoff of speed of compute time and generlize error estimator
# repeat cross validation : 3 times or more repeat to give more robust estimate.
control <- trainControl(method = "repeatedcv",number=10, repeats = 3)

metric <- "Rsquared"
# train the model - learning Vector Quantization (LVQ)
fit.rf <- train(UnixTimestamp~., data=dropNonNumericData, method="rf", metric=metric, trControl=control)
# estimate variable importance
importance <- varImp(fit.rf, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)


#######################################################
# 3. Features Selection
#######################################################
library('randomForest')
# install.packages("Metrics")
library('Metrics')
library(dplyr)

# load data
data_for_FeatureSelection = read.csv(file = '/Users/manishparihar/Desktop/ethereum2018.csv', header = TRUE)
# data_for_FeatureSelection
head(data_for_FeatureSelection)


# Rename All column name for better visibility of code
names(data_for_FeatureSelection)[names(data_for_FeatureSelection) == "Blockno"] <- "BlockNumber"
names(data_for_FeatureSelection)[names(data_for_FeatureSelection) == "UnixTimestamp"] <- "TimeStamp"
names(data_for_FeatureSelection)[names(data_for_FeatureSelection) == "Value_IN.ETH."] <- "InValueEther"
names(data_for_FeatureSelection)[names(data_for_FeatureSelection) == "Value_OUT.ETH."] <- "OutValueEther"
names(data_for_FeatureSelection)[names(data_for_FeatureSelection) == "CurrentValue....207.57.Eth"] <- "CurrentValueEther"
names(data_for_FeatureSelection)[names(data_for_FeatureSelection) == "TxnFee.ETH."] <- "TransactionFeeInEther"
names(data_for_FeatureSelection)[names(data_for_FeatureSelection) == "TxnFee.USD."] <- "TransactionFeeInUSD"
names(data_for_FeatureSelection)[names(data_for_FeatureSelection) == "Historical..Price.Eth"] <- "HistoricalPriceEther"

# Changed names
colnames(data_for_FeatureSelection)

# drop column
drop <- c("PrivateNote","ContractAddress","Status","ErrCode")
df_FeatureSelection = data_for_FeatureSelection[,!(names(data_for_FeatureSelection) %in% drop)]
head(df_FeatureSelection)

# drop unnecessary non numeric data
removeTranX_datadrop <- c("Txhash","DateTime","From","To")
dropNonNumericData_FS = df_FeatureSelection[,!(names(df_FeatureSelection) %in% removeTranX_datadrop)]
# final dataset
dropNonNumericData_FS
colnames(dropNonNumericData_FS)


set.seed(103)
dim(dropNonNumericData_FS)
# load the dataset
train <- dropNonNumericData_FS[1:2000,]
test <- dropNonNumericData_FS[2000:3000,]

# define control using a random forest selection function
# importance TRUE will give two values that is %IncMSE and IncNodePurity
model_rf<-randomForest(HistoricalPriceEther ~ ., data = train, importance=TRUE)

preds <- predict(model_rf,test[,-8])
table(preds)

# 
importance(model_rf)

# Final Model 
# varImpPlot(model_rf)
varImpPlot(model_rf,main='Variable Importance Plot',pch=16,col='black')
# Its just an outcome which tells how important is the feature  
# And how feature selection makes a difference, not only we have imporved 
# the accuracy but by using 7 predictors instead of 16.
# 1.we increased the interpretability of model
# 2. reduced the complexity of the model
# 3. reduced the training time of the model




