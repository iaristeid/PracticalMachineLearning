Coursera "Practical Machine Learning" Project
========================================================
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 



## Data 

The training data for this project are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. 


## Data Ingestion

First we download the training and test data sets and then we load them into R


```r
# setwd('./R/PracticalMachineLearning/')
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
    destfile = "./data/pml-training.csv")
```

```
## Error: unsupported URL scheme
```

```r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
    destfile = "./data/pml-testing.csv")
```

```
## Error: unsupported URL scheme
```

```r
training <- read.csv("./data/pml-training.csv")
testing <- read.csv("./data/pml-testing.csv")
```


# Feature Selection
In order to select the features to be used as predictor, we try to find the correlation between features and we locate those which have maximum impact on others. The training data set was already very sparse, with a lot of NA values.
The list of most important features is the following:


```r
M <- abs(cor(training[sapply(training, is.numeric)]))
diag(M) <- 0
which(M > 0.8, arr.ind = T)
```

```
##                  row col
## yaw_belt           7   5
## total_accel_belt   8   5
## accel_belt_y      29   5
## accel_belt_z      30   5
## accel_belt_x      28   6
## magnet_belt_x     31   6
## roll_belt          5   7
## roll_belt          5   8
## accel_belt_y      29   8
## accel_belt_z      30   8
## pitch_belt         6  28
## magnet_belt_x     31  28
## roll_belt          5  29
## total_accel_belt   8  29
## accel_belt_z      30  29
## roll_belt          5  30
## total_accel_belt   8  30
## accel_belt_y      29  30
## pitch_belt         6  31
## accel_belt_x      28  31
## gyros_arm_y       49  48
## gyros_arm_x       48  49
## magnet_arm_x      54  51
## accel_arm_x       51  54
## magnet_arm_z      56  55
## magnet_arm_y      55  56
## accel_dumbbell_x  89  67
## accel_dumbbell_z  91  68
## gyros_dumbbell_z  88  86
## gyros_forearm_z  117  86
## gyros_dumbbell_x  86  88
## gyros_forearm_z  117  88
## pitch_dumbbell    67  89
## yaw_dumbbell      68  91
## gyros_forearm_z  117 116
## gyros_dumbbell_x  86 117
## gyros_dumbbell_z  88 117
## gyros_forearm_y  116 117
```


## Preparation for Cross-Validation

In order to measure our model we partition our training set into two parts (75%, 25%)


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
inTrain <- createDataPartition(training$classe, p = 0.75, list = FALSE)
train1 <- training[inTrain, ]
test1 <- training[-inTrain, ]
```



## Prediction Models

We will be using Random Forest based on the first part of training set. 
We will compare the two options: 

- random forest of caret package
- random forest of randomForest package

We will estimate errors.
randomForest packages returns the prediction error in the results.
For caret package, we will compare predictions towards the classe values of the second part of training set.


```r
library(caret)
library(randomForest)

rf1 <- train(classe ~ user_name + roll_belt + pitch_belt + yaw_belt + gyros_belt_x + 
    gyros_belt_y + gyros_belt_z + accel_belt_x + accel_belt_y + accel_belt_z + 
    magnet_belt_x + magnet_belt_y + magnet_belt_z + roll_arm + pitch_arm + yaw_arm + 
    total_accel_arm + gyros_arm_x + gyros_arm_y + gyros_arm_z + accel_arm_x + 
    accel_arm_y + accel_arm_z + magnet_arm_x + magnet_arm_y + magnet_arm_z, 
    train1, method = "rf")

pred1 <- predict(rf1, test1[, -160])

A <- table(pred1, test1$classe)
error1 <- (sum(A) - sum(diag(A)))/sum(A)

predRight1 <- pred1 == test1$classe


rf2 <- randomForest(formula = classe ~ user_name + roll_belt + pitch_belt + 
    yaw_belt + gyros_belt_x + gyros_belt_y + gyros_belt_z + accel_belt_x + accel_belt_y + 
    accel_belt_z + magnet_belt_x + magnet_belt_y + magnet_belt_z + roll_arm + 
    pitch_arm + yaw_arm + total_accel_arm + gyros_arm_x + gyros_arm_y + gyros_arm_z + 
    accel_arm_x + accel_arm_y + accel_arm_z + magnet_arm_x + magnet_arm_y + 
    magnet_arm_z, data = train1, ntree = 500)

importance(rf2)
```

```
##                 MeanDecreaseGini
## user_name                  234.8
## roll_belt                 1396.6
## pitch_belt                 993.9
## yaw_belt                  1241.0
## gyros_belt_x               188.0
## gyros_belt_y               145.4
## gyros_belt_z               359.7
## accel_belt_x               224.0
## accel_belt_y               158.8
## accel_belt_z               526.0
## magnet_belt_x              397.4
## magnet_belt_y              465.0
## magnet_belt_z              517.7
## roll_arm                   610.9
## pitch_arm                  432.5
## yaw_arm                    381.9
## total_accel_arm            193.8
## gyros_arm_x                318.6
## gyros_arm_y                316.1
## gyros_arm_z                160.1
## accel_arm_x                412.8
## accel_arm_y                336.9
## accel_arm_z                364.0
## magnet_arm_x               448.4
## magnet_arm_y               427.6
## magnet_arm_z               383.3
```

```r

rf2
```

```
## 
## Call:
##  randomForest(formula = classe ~ user_name + roll_belt + pitch_belt +      yaw_belt + gyros_belt_x + gyros_belt_y + gyros_belt_z + accel_belt_x +      accel_belt_y + accel_belt_z + magnet_belt_x + magnet_belt_y +      magnet_belt_z + roll_arm + pitch_arm + yaw_arm + total_accel_arm +      gyros_arm_x + gyros_arm_y + gyros_arm_z + accel_arm_x + accel_arm_y +      accel_arm_z + magnet_arm_x + magnet_arm_y + magnet_arm_z,      data = train1, ntree = 500) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 5
## 
##         OOB estimate of  error rate: 1.68%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4149   14   10   10    2    0.008602
## B   26 2805   14    2    1    0.015098
## C    5   68 2476   16    2    0.035450
## D    7    3   41 2357    4    0.022803
## E    2    9    7    4 2684    0.008130
```

```r

pred2 <- predict(rf1, test1[, -160])

B <- table(pred2, test1$classe)
error2 <- (sum(B) - sum(diag(B)))/sum(B)

predRight2 <- pred2 == test1$classe

```



## Final Model
For the final prediction we will apply random forest on the totality of the training set.
We will keep the ntree to 500 (due to computing power limitations).


```r
rf3 <- randomForest(formula = classe ~ user_name + roll_belt + pitch_belt + 
    yaw_belt + gyros_belt_x + gyros_belt_y + gyros_belt_z + accel_belt_x + accel_belt_y + 
    accel_belt_z + magnet_belt_x + magnet_belt_y + magnet_belt_z + roll_arm + 
    pitch_arm + yaw_arm + total_accel_arm + gyros_arm_x + gyros_arm_y + gyros_arm_z + 
    accel_arm_x + accel_arm_y + accel_arm_z + magnet_arm_x + magnet_arm_y + 
    magnet_arm_z, data = training, na.action = na.omit, ntree = 500)

importance(rf3)
```

```
##                 MeanDecreaseGini
## user_name                  328.2
## roll_belt                 1869.2
## pitch_belt                1319.3
## yaw_belt                  1708.6
## gyros_belt_x               246.7
## gyros_belt_y               191.4
## gyros_belt_z               481.2
## accel_belt_x               286.3
## accel_belt_y               193.2
## accel_belt_z               728.2
## magnet_belt_x              525.1
## magnet_belt_y              608.9
## magnet_belt_z              691.7
## roll_arm                   822.6
## pitch_arm                  587.2
## yaw_arm                    502.4
## total_accel_arm            250.5
## gyros_arm_x                426.8
## gyros_arm_y                422.7
## gyros_arm_z                208.1
## accel_arm_x                551.8
## accel_arm_y                428.7
## accel_arm_z                475.0
## magnet_arm_x               593.9
## magnet_arm_y               560.2
## magnet_arm_z               505.4
```


## Final Outcome 

The prediction outcome of the final model on the original test set is:


```r
predict(rf3, testing)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


