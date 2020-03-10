# @title:  UnSupervised Learning Algorithms - K-Mean Clustering and K-Medoids
# @author: Manish Parihar
# @date:   05/01/2020

set.seed(100)
# 1. load data of ethereum
data_2018 = read.csv(file ='/Users/manishparihar/Desktop/Infura/ethereum2018.csv', header = TRUE)
data_2018

# drop particular complete column from database
drop <- c("PrivateNote","ContractAddress","Status","ErrCode")
df = data_2018[,!(names(data_2018) %in% drop)]
df

# drop unnecessary non numeric data
removeTranX_datadrop <- c("Txhash","DateTime","From","To")
dropNonNumericData = df[,!(names(df) %in% removeTranX_datadrop)]
# final dataset
dropNonNumericData

# install cluster library for clustering algorithms
# install.packages('corrplot')
# factoextra for visualization
# install.packages('factoextra')
# install.packages("tibble")
library(corrplot)
library(cluster)
library(factoextra)
library(magrittr)
library(tibble)
library(dplyr)
library(gridExtra)

# Summary statistics of dropNonNumericData
summary(dropNonNumericData)
# Show all column names
colnames(dropNonNumericData)

# Rename All column name for better understanding of code
names(dropNonNumericData)[names(dropNonNumericData) == "Blockno"] <- "BlockNumber"
names(dropNonNumericData)[names(dropNonNumericData) == "UnixTimestamp"] <- "TimeStamp"
names(dropNonNumericData)[names(dropNonNumericData) == "Value_IN.ETH."] <- "InValueEther"
names(dropNonNumericData)[names(dropNonNumericData) == "Value_OUT.ETH."] <- "OutValueEther"
names(dropNonNumericData)[names(dropNonNumericData) == "CurrentValue....207.57.Eth"] <- "CurrentValueEther"
names(dropNonNumericData)[names(dropNonNumericData) == "TxnFee.ETH."] <- "TransactionFeeInEther"
names(dropNonNumericData)[names(dropNonNumericData) == "TxnFee.USD."] <- "TransactionFeeInUSD"
names(dropNonNumericData)[names(dropNonNumericData) == "Historical..Price.Eth"] <- "HistoricalPriceEther"

# Changed names of column
colnames(dropNonNumericData)

# Correlation Plot of data
corrmatrix <- cor(dropNonNumericData)
# Positive correlations are displayed in blue and negative correlations in red color. 
# Color intensity and the size of the circle are proportional to the correlation coefficients.
# In the plot, correlations with p-value > 0.01 are considered as insignificant. 
# In this case the correlation coefficient values are leaved blank or crosses are added.
corrplot(corrmatrix, method = 'number')

# remove first two column Blockno and UnixTimeStamp which is not useful
attribute_six_data <- dropNonNumericData[-c(1,2)]

# top 3 row count from selected database
head(attribute_six_data,3)

# -------------------- Experiment 1 - Silhouette -------------------- #
 
# The above method of calculating silhouette score using silhouette() and 
# plotting the results states that optimal number of clusters as 2
# generate nstart = 25 initial configurations. (recommended approach)

km <- kmeans(attribute_six_data, centers = 2, nstart=25)
str(km)

silhouette_score <- function(k){
  km <- kmeans(attribute_six_data, centers = k, nstart=25)
  ss <- silhouette(km$cluster, dist(attribute_six_data))
  mean(ss[, 3])
}

k <- 2:10
avg_sil <- sapply(k, silhouette_score)
avg_sil
plot(k, type='b', avg_sil, xlab='Number of clusters', ylab='Average Silhouette Scores', frame=FALSE)

# The other method with visual aid is using factoextra package
fviz_nbclust(attribute_six_data, kmeans, method='silhouette')
# Output == The optimal number of clusters is 6.

# Actual Clustering
km.final <- kmeans(attribute_six_data, 6)
## Total Within cluster sum of square
km.final$tot.withinss
## Cluster sizes
km.final$size

dropNonNumericData$cluster <- km.final$cluster
head(dropNonNumericData, 6)

# Percentage accuray of data
print(km.final, 6)
# Within cluster sum of squares by cluster:
#  (between_SS / total_SS =  99.03339 %)
km.final$betweenss/km.final$totss*100

# vizualization of cluster
fviz_cluster(km.final, data=attribute_six_data)

dropNonNumericData %>%
  as_tibble() %>%
  mutate(cluster = km.final$cluster,
         state = row.names(dropNonNumericData)) %>%
  ggplot(aes(InValueEther, OutValueEther, color = factor(cluster), label = state)) +
  geom_text()


dropNonNumericData %>%
  mutate(Cluster = km.final$cluster) %>%
  group_by(Cluster) %>%
  summarise_all("mean")
# k from 2 to 6 cluster
k2 <- kmeans(attribute_six_data, centers = 2, nstart=25)
k3 <- kmeans(attribute_six_data, centers = 3, nstart = 25)
k4 <- kmeans(attribute_six_data, centers = 4, nstart = 25)
k5 <- kmeans(attribute_six_data, centers = 5, nstart = 25)
k6 <- kmeans(attribute_six_data, centers = 6, nstart = 25)


# plots to compare all six clusters
p1 <- fviz_cluster(k2, geom = "point", data = dropNonNumericData) + ggtitle("k = 2")
p2 <- fviz_cluster(k3, geom = "point",  data = dropNonNumericData) + ggtitle("k = 3")
p3 <- fviz_cluster(k4, geom = "point",  data = dropNonNumericData) + ggtitle("k = 4")
p4 <- fviz_cluster(k5, geom = "point",  data = dropNonNumericData) + ggtitle("k = 5")
p5 <- fviz_cluster(k6, geom = "point",  data = dropNonNumericData) + ggtitle("k = 6")

# grid plot
grid.arrange(p1, p2, p3, p4,p5, nrow = 2)


# -------------------- Experiment 2 - Elbow Method -------------------- #

# install.packages("tidyverse")
# install.packages("purrr")
library(tidyverse)
library(purrr)

# function we use to compute total within-cluster sum of square
wss <- function(k){
  kmeans(attribute_six_data, k, nstart = 10)$tot.withinss
}

# Compare and plot wss for  k = 1 to k = 15
k.values <- 1:15

# extract wss for 2 to 15 clusters
wss_values <- map_dbl(k.values,wss)
# plot
plot(k.values, wss_values, type = "b", pch=19, frame = FALSE,
     xlab = "Number of clusters K",
     ylab = "Total within-clusters sum of squares")

# plot
fviz_nbclust(attribute_six_data, kmeans, method = "wss")

# Percentage accuray of data
print(km.final, 4)
km.final$betweenss/km.final$totss*100
# Result got as 99 %

# --------------------  Experiment 3 - Principal Component -------------------- #
# install.packages("ggfortify")
library(ggfortify)
library(RColorBrewer)

df <- dropNonNumericData
df
autoplot(prcomp(dropNonNumericData))
autoplot(prcomp(dropNonNumericData), data = data_2018, color=Species)
#autoplot(prcomp(dropNonNumericData), data = data_2018, color=Species, label = TRUE, label.size = 3)
autoplot(prcomp(df), data = data_2018, color=Species, shape = FALSE, label.size = 3)

autoplot(prcomp(df), data = data_2018, color=Species, loadings = TRUE)

autoplot(prcomp(df), data = data_2018, color=Species,
         loadings = TRUE, loadings.colour = 'blue',
         loadings.label = TRUE, loadings.label.size = 3)

#install.packages("ISLR")
library(ISLR)
data(dropNonNumericData)
apply(dropNonNumericData, 2, mean)
apply(dropNonNumericData, 2, sd)

dropNonNumericData_pca = prcomp(dropNonNumericData,scale = TRUE)
names(dropNonNumericData_pca)
summary(dropNonNumericData_pca)

dropNonNumericData_pca$center
dropNonNumericData_pca$scale
dropNonNumericData_pca$rotation
dim(dropNonNumericData_pca$x)
dim(dropNonNumericData)
head(dropNonNumericData_pca$x)

scale(as.matrix(dropNonNumericData))[1, ] %*% dropNonNumericData_pca$rotation[, 1]

# plot
biplot(dropNonNumericData_pca,scale = 0, cex = 0.5)

dropNonNumericData_pca$sdev
dropNonNumericData_pca$sdev ^ 2 / sum(dropNonNumericData_pca$sdev ^ 2)

get_PVE = function(pca_out) {
  pca_out$sdev ^ 2 / sum(pca_out$sdev ^ 2)
}

pve = get_PVE(dropNonNumericData_pca)
pve

plot(
  pve,
  xlab = "Principal Component",
  ylab = "Proportion of Variance Explained",
  ylim = c(0, 1),
  type = 'b'
)

# We can then plot the proportion of variance explained for each PC. 
# As expected, we see the PVE decrease.
cumsum(pve)

plot(
  cumsum(pve),
  xlab = "Principal Component",
  ylab = "Cumulative Proportion of Variance Explained",
  ylim = c(0, 1),
  type = 'b'
)

summary(pve)

# For data % in variance
dropsumm <- prcomp(dropNonNumericData)
summary(dropsumm)
# 94% Variance

#-------------------- Experiment 4 - K-Mediods(Partitioning Around Medoids)  -------------------- #
df_kmediods <- dropNonNumericData
df_kmediods

library(cluster)
library(factoextra)

# PAM Algorithm with k = 2
pam.result <- pam(df_kmediods, 2)
print(pam.result)

# add point classification to the original dataset
datapoint_add <- cbind(data_2018, cluster = pam.result$cluster)
head(datapoint_add, n = 3)

# pam() returns an object of class pam:
# with component - medoids, clustering
# Cluster medoids :
pam.result$medoids

# cluster numbers
head(pam.result$clustering)

# Plot PAM Clusters
fviz_cluster(pam.result)
# Output k-Medoids algorithm, PAM is a robust alternative methods to k-means
# for partitioning a data set into clusers of observations. Each cluster represented
# by selected object within the cluster. these selected objects are named medoids and 
