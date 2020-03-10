# @title:  Supervised Learning Algorithms - K-Mean Clustering #
# @author: Manish Parihar
# @date:   31/01/2020
###################    SVM - Classification Algorithm  ###################
###################    Experiment on  Dataset   ###################
set.seed(202)
library(dplyr)

# install.packages("caTools")
library(caTools)

# # import data
# data_Address = read.csv(file = '/Users/manishparihar/Desktop/Thesis_Data/FinalScamDB.csv', header = TRUE)
# 
# data_Address["Subcategory_Filter"] <- NA
# 
# data_Address$address[data_Address$address == "Nan"] <- 0
# data_Address$address
# 
# 
# 
# # Selected Dataset
# data_Address = data_Address[4:5] 
# 
# dim(data_Address)
# 
# # encode the target feature as factor
# data_Address$malicious = factor(data_Address$malicious, levels = c(0,1))
# 
# 
# # split the dataset into training and test dataset into 80 and 20
# split = sample.split(data_Address$malicious, SplitRatio = 0.8)
# split
# 
# training_set = subset(data_Address$malicious, split==TRUE)
# test_set = subset(data_Address$malicious, split==FALSE)
# 
# # Feature Scaling
# training_set[-3] = scale(training_set[-3]) 
# test_set[-3] = scale(test_set[-3]) 
# 
# head(training_set)
# 
# #install.packages('e1071')
# library(e1071)
# classifierL = svm(formula = malicious ~ .,
#                   data = training_set,
#                   type = 'C-classification',
#                   kernel = 'linear')


########################################################
# Data Prep

etherum10k <- read.csv("~/Desktop/Infura/etherum10k.csv", sep=";")
etherum10k

myDataframe = data.frame(etherum10k)


# select 
#cumulativeGas,gasUsed,confirmations,isError,gasPrice,gas
myDataframe  = myDataframe[10:12] 
myDataframe

# Encoding
myDataframe$isError = factor(myDataframe$isError, levels = c(0, 1)) 

# split the dataset into training and test dataset into 80 and 20
split = sample.split(myDataframe$isError, SplitRatio = 0.8)
split

training_set = subset(myDataframe, split==TRUE)
test_set = subset(myDataframe, split==FALSE)

# Feature Scaling
training_set[-3] = scale(training_set[-3]) 
test_set[-3] = scale(test_set[-3]) 

head(training_set)

#install.packages('e1071')
library(e1071)
classifier = svm(formula = isError ~ .,
                  data = training_set,
                  type = 'C-classification',
                  kernel = 'linear')

# Predict the test set results
y_prediction = predict(classifier, newdata = test_set[-3]) 
y_prediction

# Create Confusion Matrix 
confusionMat = table(test_set[, 3], y_prediction) 
confusionMat

# installing library ElemStatLearn 
# install.packages("ElemStatLearn")
#library(ElemStatLearn) 

set = training_set 
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01) 
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01) 

grid_set = expand.grid(X1, X2) 
colnames(grid_set) = c('gas', 'gasPrice') 
y_grid = predict(classifier, newdata = grid_set) 

plot(set[, -3], 
     main = 'SVM (Training set)', 
     xlab = 'Gas', ylab = 'Gas Price', 
     xlim = range(X1), ylim = range(X2)) 

contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE) 

points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'coral1', 'aquamarine')) 

points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Test Vizualisation
set = test_set 
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01) 
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01) 

grid_set = expand.grid(X1, X2) 
colnames(grid_set) = c('gas', 'gasPrice') 
y_grid = predict(classifier, newdata = grid_set) 

plot(set[, -3], main = 'SVM (Test set)', 
     xlab = 'Gas', ylab = 'Gas Price', 
     xlim = range(X1), ylim = range(X2)) 

contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE) 

points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'coral1', 'aquamarine')) 

points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3')) 



# four fold plot
# fourfoldplot(table_mat)
fourfoldplot(confusionMat,
             color = c("#B22222", "#2E8B57"),
             main = "FourFold Plot")
