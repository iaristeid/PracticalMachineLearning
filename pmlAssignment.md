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

First we download the training and test data sets and then we load them into R (training and testing data frames).


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
In order to select the features to be used as predictors, we calculate the correlation between features and we locate those which have maximum impact on others (absolute value of correlation > 0.8)
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

In order to measure our model we partition our training set into two parts (75%, 25%): train1, test1 data frames.


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

We will be using Random Forest based on the first part of our training set (train1). 


```r
library(caret)

rf1 <- train(classe ~ user_name + roll_belt + pitch_belt + yaw_belt + gyros_belt_x + 
    gyros_belt_y + gyros_belt_z + accel_belt_x + accel_belt_y + accel_belt_z + 
    magnet_belt_x + magnet_belt_y + magnet_belt_z + roll_arm + pitch_arm + yaw_arm + 
    total_accel_arm + gyros_arm_x + gyros_arm_y + gyros_arm_z + accel_arm_x + 
    accel_arm_y + accel_arm_z + magnet_arm_x + magnet_arm_y + magnet_arm_z, 
    train1, method = "rf")
```


## Error Estimation

We will estimate errors by comparing predictions towards the real classe values of the second part of training set (test1).
Error probability and number of correct predictions are displayed below:


```r
pred1 <- predict(rf1, test1[, -160])
A <- table(pred1, test1$classe)
error1 <- (sum(A) - sum(diag(A)))/sum(A)
predRight1 <- pred1 == test1$classe

error1
```

```
## [1] 0.01815
```

```r
sum(predRight1)
```

```
## [1] 4815
```



## Final Prediction Outcome

The prediction outcome of the final model on the original test set is:


```r
predict(rf1, testing)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


