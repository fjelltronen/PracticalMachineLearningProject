
training<-read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
testing<-read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")

## not all windows have a new_window
table(training$num_window,training$new_window)

## use user information?
table(training$user_name,training$classe)
# may use user as a feature

## removing unwanted columns:
toRemove <- c("X","raw_timestamp_part_1","user_name", "raw_timestamp_part_2","cvtd_timestamp","new_window","num_window")
training<-training[, !(names(training) %in% toRemove)]

set.seed(2018-5-12)

library(caret)
## identify features that have no variability
nzVar <- nearZeroVar(training, saveMetrics=TRUE)
# keep only those with nzv == TRUE
toKeep <- row.names(nzVar[nzVar$nzv == FALSE,])
training<-training[, (names(training) %in% toKeep)]


## preProcessing object to handle missing data
preObj <- preProcess(training[,-94],method="knnImpute")
trainingPP <- predict(preObj, training[,-94])

## preProcessing object to select features
#preObj <- preProcess(log10(trainingPP+1),method="pca")
#trainingPC <- predict(preObj, training[,-94])

## training options
# fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 2)
# training
#fit <- train(training$classe ~ ., data = trainingPP, method = "rf", trControl = fitControl, ntrees = 10, do.trace = TRUE)
fit <- train(training$classe ~ ., data = trainingPP, method = "rf", 
             trControl = trainControl(method="cv"), ntrees = 10, do.trace = TRUE)

