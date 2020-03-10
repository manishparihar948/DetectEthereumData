# @title:  Supervised Learning Algorithms - K-Mean Clustering #
# @author: Manish Parihar
# @date:   31/01/2020
###################    Final Experiment  ##############################################
###################    1. Experiment on First Dataset Ethereum10k Dataset   ##############
data_Address <- read.csv("~/Desktop/Infura/etherum10k.csv", sep=";")

#data_Address
dim(data_Address)
library(rpart.plot)

data(data_Address)
# 80% of the sample size
smp_size <- floor(0.8 * nrow(data_Address))
train_datAddress <- sample(seq_len(nrow(data_Address)), size = smp_size)

train_dataAddress <- data_Address[train_datAddress, ]

test_dataAddress <- data_Address[-train_datAddress, ]

# prop.table() combine with table() to verify if the randomization process is correct.
prop.table(table(train_dataAddress$isError))
prop.table(table(test_dataAddress$isError))

# Build model for classification
# method : class for classification, anova for regression

library(rpart)
fit <- rpart(isError~., data = train_dataAddress, method = 'class')

# Make prediction
# prediction on test data 
# type of prediction : 'class' for classification, 'prob' for probability
predict_unseen <-predict(fit, test_dataAddress, type = 'class')
predict_unseen

# Testing malicious activity, who is malicious and who is non malicious
table_mat <- table(test_dataAddress$isError, predict_unseen)
table_mat

# accuracy test
accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
accuracy_Test
print(paste('Accuracy for test', accuracy_Test))


# Confusion Matrix 
confuse_mat <-confusionMatrix(table(test_dataAddress$isError, predict_unseen))
confuse_mat

# Tune hyper parameter
accuracy_tune <- function(fit) {
  predict_unseen <- predict(fit, test_dataAddress, type = 'class')
  table_mat <- table(test_dataAddress$isError, predict_unseen)
  accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
  accuracy_Test
}

control <- rpart.control(minsplit = 4,
                         minbucket = round(5 / 3),
                         maxdepth = 3,
                         cp = 0)
tune_fit <- rpart(isError~., data = train_dataAddress, method = 'class', control = control)

accuracy_tune(tune_fit)

# four fold plot
# fourfoldplot(table_mat)
fourfoldplot(table_mat,
             color = c("#B22222", "#2E8B57"),
             main = "FourFold Plot")



# plot(performance(pred, "acc"))

###################    Decision Tree    ###################
# Step 1. Clean the dataset

set.seed(201)
data_DT = read.csv(file = '/Users/manishparihar/Desktop/Thesis_Data/DarkAddressList.csv', header = TRUE)
#data_DT
dim(data_DT)

# Step 2. Clean the dataset

# drop column
removeX_fromData <- c("X")
dropUnusedData = data_DT[,!(names(data_DT) %in% removeX_fromData)]
# final dataset
# dropUnusedData
dim(dropUnusedData)


# returns TRUE of x is missing
is.na(dropUnusedData) 
# create new dataset without missing data
newdata <- na.omit(dropUnusedData)

# Step 3. Create train/test set
# install.packages("rpart.plot")
library(rpart.plot)

# ====================#############==================== #
data(dropUnusedData)
# 80% of the sample size
smp_size <- floor(0.8 * nrow(dropUnusedData))
train_dat <- sample(seq_len(nrow(dropUnusedData)), size = smp_size)

train_data <- dropUnusedData[train_dat, ]

test_data <- dropUnusedData[-train_dat, ]
dim(dropUnusedData)
colnames(dropUnusedData)

# prop.table() combine with table() to verify if the randomization process is correct.
prop.table(table(train_data$address))
prop.table(table(test_data$address))

# build model
library(rpart)
fit <- rpart(date~., data = train_data, method = 'anova')
predict_unseen <-predict(fit, test_data, type = 'prob')
predict_unseen

table_mat <- table(test_data$date, predict_unseen)
table_mat

accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
accuracy_Test
print(paste('Accuracy for test', accuracy_Test))


accuracy_tune <- function(fit) {
  predict_unseen <- predict(fit, test_data, type = 'vector')
  table_mat <- table(test_data$date, predict_unseen)
  accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
  accuracy_Test
}


control <- rpart.control(minsplit = 4,
                         minbucket = round(5 / 3),
                         maxdepth = 3,
                         cp = 0)
tune_fit <- rpart(date~., data = train_data, method = 'anova', control = control)
accuracy_tune(tune_fit)

##################################################################################
###################   2. Experiment on Dataset  FinalScamDB Dataset   ############
data_Address = read.csv(file = '/Users/manishparihar/Desktop/Thesis_Data/FinalScamDB.csv', header = TRUE)
#data_Address
dim(data_Address)
library(rpart.plot)

data(data_Address)
# 80% of the sample size
smp_size <- floor(0.8 * nrow(data_Address))
train_datAddress <- sample(seq_len(nrow(data_Address)), size = smp_size)

train_dataAddress <- data_Address[train_datAddress, ]

test_dataAddress <- data_Address[-train_datAddress, ]

# prop.table() combine with table() to verify if the randomization process is correct.
prop.table(table(train_dataAddress$malicious))
prop.table(table(test_dataAddress$malicious))

# Build model for classification
# method : class for classification, anova for regression

library(rpart)
fit <- rpart(malicious~., data = train_dataAddress, method = 'class')

# Make prediction
# prediction on test data 
# type of prediction : 'class' for classification, 'prob' for probability
predict_unseen <-predict(fit, test_dataAddress, type = 'class')
predict_unseen

# Testing malicious activity, who is malicious and who is non malicious
table_mat <- table(test_dataAddress$malicious, predict_unseen)
table_mat

# accuracy test
accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
accuracy_Test
print(paste('Accuracy for test', accuracy_Test))


# Confusion Matrix 
confuse_mat <-confusionMatrix(table(test_dataAddress$malicious, predict_unseen))
confuse_mat

# four fold plot
fourfoldplot(table_mat)

###### Final Result ################################################
#   Confusion Matrix and Statistics
# 
# predict_unseen
# 0    1
# 0 1516    0
# 1   20    1
# 
# Accuracy : 0.987        
# 95% CI : (0.98, 0.992)
# No Information Rate : 0.9993       
# P-Value [Acc > NIR] : 1            
# 
# Kappa : 0.0898       
# 
# Mcnemar's Test P-Value : 2.152e-05    
#                                        
#             Sensitivity : 0.98698      
#             Specificity : 1.00000      
#          Pos Pred Value : 1.00000      
#          Neg Pred Value : 0.04762      
#              Prevalence : 0.99935      
#          Detection Rate : 0.98634      
#    Detection Prevalence : 0.98634      
#       Balanced Accuracy : 0.99349      
#                                        
#        'Positive Class' : 0   
#        
#       


