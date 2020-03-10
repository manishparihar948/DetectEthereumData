# @title:  Supervised Learning Algorithms - K-Mean Clustering #
# @author: Manish Parihar
# @date:   31/01/2020

##############################################################################################################
############################################ NaiveBayes Algorithm ############################################
##############################################################################################################

# Experiment on NaiveBayes
#Loading required packages
#install.packages('tidyverse')
 library(tidyverse)
#install.packages('ggplot2')
library(ggplot2)
#install.packages('caret')
library(caret)
#install.packages('caretEnsemble')
library(caretEnsemble)
#install.packages('psych')
library(psych)
#install.packages('Amelia')
library(Amelia)
#install.packages('mice')
library(mice)
#install.packages('GGally')
library(GGally)
#install.packages('rpart')
library(rpart)
#install.packages('randomForest')
library(randomForest)
library(tidyverse)
library(dplyr)
 
# load dataset
etherum10k <- read.csv("~/Desktop/Infura/etherum10k.csv", sep=";")
etherum10k


# remove contract address column
etherum10k$contractAddress <- NULL
etherum10k


nv_dataframe = as_tibble(etherum10k)
nv_dataframe

# select all numeric 
new_df <- nv_dataframe %>% select_if(is.numeric)
new_df

new_df <- data.frame(new_df)
new_df <- new_df[-1,]
new_df

#Setting outcome variables as categorical
new_df$isError <- factor(new_df$isError, levels = c(0,1), labels = c("False", "True"))

# studying structure of data
str(new_df)
head(new_df)

# describe data
describe(new_df)

# check missing data in plot
missmap(new_df)

# remove hash, blockhash, from, to, value, gasPrice
new_df = new_df[,-4]
new_df = new_df[,-(5:7)]
new_df = new_df[,-6]
# remove missing value 
# delete column 'contractAddress' as it was unused and empty
#new_df <- select(new_df, -contractAddress)
#new_df


# check missing data in plot
# missmap(new_df)


#visualisation
P <- ggplot(new_df, aes(x=gasUsed, fill=isError, color=isError)) +
  geom_histogram(binwidth = 10) + labs(title="Naive Bayes: Error in Transaction")
P + theme_bw()

#Data Visualization
#Visual 1
ggplot(new_df, aes(gasUsed, colour = isError)) +
  geom_freqpoly(binwidth = 1) + labs(title="Transaction with an error")



# Data Modelling
# Build model
# split data into training and test datasets
nv_Train <- createDataPartition(y = new_df$isError,p = 0.8,list = FALSE)
training_data <- new_df[nv_Train,]
testing_data <- new_df[-nv_Train,]

#Check dimensions of the split
prop.table(table(new_df$isError)) * 100


# for training
prop.table(table(training_data$isError)) * 100

# for testing
prop.table(table(testing_data$isError)) * 100

# for comparing error of the training and test phase
# now lets create separate variables that store value of response variable
# create objects x  holds predictor variables 
# y hold error variable
# we removes factor based feature/attribute
x = training_data[,-6]

y = training_data$isError


library(e1071)
library(klaR)
library(MASS)

# Now Create Naive Bayes model by using training dataset

model = train(x,y,'nb',trControl=trainControl(method='cv',number=10))
model

#Plot Variable performance
X <- varImp(model,scale=F)
plot(X)

# ggparis
ggpairs(x)

# Model Evaluation
# predict test set
Predict <- predict(model,newdata = testing_data )

# confusion matrix to see accuracy value and other parameter values
confusionMat <- confusionMatrix(Predict, testing_data$isError)
confusionMat


# Plot
TClass <- factor(c(0, 0, 1, 1))
PClass <- factor(c(0, 1, 0, 1))
Y      <- c(31, 1687, 247, 34)
df <- data.frame(TClass, PClass, Y)

library(ggplot2)
ggplot(data =  df, mapping = aes(x = TClass, y = PClass)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "lightgrey", high = "lightblue") +
  theme_bw() + theme(legend.position = "none")


##############################################################################################################
############################################ K-NN Algorithm ##################################################
##############################################################################################################

# load dataset
etherum10k <- read.csv("~/Desktop/Infura/etherum10k.csv", sep=";")
# etherum10k

# remove contract address column
etherum10k$contractAddress <- NULL
# etherum10k

# select all numeric 
# new_df <- etherum10k %>% select_if(is.numeric)
# new_df

new_df <- data.frame(etherum10k)
new_df <- new_df[-1,]
new_df

# dimension of data
dim(new_df)
# type of data
typeof(new_df)
# data type of all attributes
str(new_df)

# data summary
#summary(new_df)

# remove 7th column 'to'
# remove hash, blockhash, from, to, value, gasPrice

new_df <- new_df[,-(7:9)] 

new_df <- new_df[,-5]

new_df <- new_df[,-3]

new_df <- new_df[,-6]

# remove transaction column
new_df <- new_df[,-7]

# final db
new_df

str(new_df)

# Split data in Training and Test
set.seed(301)

# save knn final dataset to rda file
save(new_df, file = "knn_new_df.rda")

# Cumulative Gas Used is the total gas used in a contract over time
# Variable "Gas Used" is my target variable , this variable will find
# whether gas used on this 6 attributes.

new_df.subset <- new_df[c('nonce','transactionIndex','gas','isError','cumulativeGasUsed',
                          'gasUsed')]


normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) } # creating a normalize function for easy convertion.

head(new_df.subset)

# lapply creates list that is why it is converted to dataframe and it applies defined fundtion (which is 'normalize') 
#to all the list values which is here column 2 to 8 as first column is target/response.
new_df.subset.n<- as.data.frame(lapply(new_df.subset[,-4], normalize)) 
head(new_df.subset.n)

#random selection of 80% data.
dat.d <- sample(1:nrow(new_df.subset.n),size=nrow(new_df.subset.n)*0.8,replace = FALSE)
dat.d

# 80% training data
train.gc <- new_df.subset[dat.d,] 

# remaining 20% test data
test.gc <- new_df.subset[-dat.d,] 

# creating seperate dataframe for 'isError' attribute
# that is our target.
train.gc_lab <- new_df.subset[dat.d,4]
train.gc_lab
test.gc_lab  <- new_df.subset[-dat.d,4]  
test.gc_lab 

# Train a model on data
# knn function call
library(class)

# total number of observation
NROW(train.gc_lab)

# identify optimal value for k,
# as we cannot take square root of nrow coz NRow = 7999
# its square root is approx 89.9 which is too hig for rstudio 
# and for calculate, it will abort the rstudio
# so assume to decide k = 31, 32
knn.30 <-  knn(train=train.gc, test=test.gc, cl=train.gc_lab, k=30)
knn.31 <-  knn(train=train.gc, test=test.gc, cl=train.gc_lab, k=31)

# Evaluate the model performance
# For knn = 30
# Accuracy is 96.85 %
ACC.30 <- 100 * sum(test.gc_lab == knn.30)/NROW(test.gc_lab) 
ACC.30
# For knn = 31
# Accuracy is 96.9 % which is increased
ACC.31 <- 100 * sum(test.gc_lab == knn.31)/NROW(test.gc_lab)  
ACC.31  

# check prediction against actual value in table
# test.gc_lab
# knn.30    0    1
# 0 1687   58
# 1    6  249
table(knn.30 ,test.gc_lab) 


# table(knn.31 ,test.gc_lab) 
# test.gc_lab
# knn.31    0    1
# 0 1687   56
# 1    6  251
table(knn.31 ,test.gc_lab) 

library(caret)
# Accuracy by confusion matrix
confusionMatrix(knn.30,as.factor(test.gc_lab))
confusionMatrix(knn.31,as.factor(test.gc_lab))


# Confusion plot
TClass <- factor(c(0, 0, 1, 1))
PClass <- factor(c(1, 0, 1, 0))
Y      <- c(1699, 4, 61, 236)
df <- data.frame(TClass, PClass, Y)

library(ggplot2)
ggplot(data =  df, mapping = aes(x = TClass, y = PClass)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_bw() + theme(legend.position = "none")


# Improve the performance of model
# tuning parameter 'k' and number of features/attributes selection

i=1                          # declaration to initiate for loop
k.optm=1                     # declaration to initiate for loop
for (i in 1:32){ 
  knn.mod <-  knn(train=train.gc, test=test.gc, cl=train.gc_lab, k=i)
  k.optm[i] <- 100 * sum(test.gc_lab == knn.mod)/NROW(test.gc_lab)
  k=i  
  cat(k,'=',k.optm[i],'\n')       # to print % accuracy 
}

# plot graph
# Maximum accuracy at k=1 but its initial condition so we avoid this value and select 
# so best value we got on 14 also
plot(k.optm, type="b", xlab="K- Value",ylab="Accuracy level")  # to plot % accuracy wrt to k-value


## Final interpretation is - At K = 1, maximum accuracy we got is 99.5, after this, 
# it looks increasing K decreases the classification but increase succes rate.
# Further accuracy cannot increased by optimising feature selection and repeating the algorithm.

