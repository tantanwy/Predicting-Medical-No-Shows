library(Hmisc)
library(rlang)
library(ggplot2)
library(Hmisc)
library(corrplot, quietly = TRUE)
library(igraph)
library(rpart, quietly = TRUE)
library(grid)
library(kernlab, quietly = TRUE)
library(nnet, quietly = TRUE)
library(sampling)
library(plyr)
library(dplyr)
library(rpart.plot)
library(caret)
library(xgboost)
library(DMwR)

crs <- new.env()
fname <- "C:/Users/ohrtwy/Desktop/WY/z. School/Modules/EB5205 Clinical/Journal/Appt No Show/Dataset/Kaggle_WY.csv"
fname_kaggle <- "C:/Users/ohrtwy/Desktop/WY/z. School/Modules/EB5205 Clinical/Journal/Appt No Show/Dataset/KaggleV2-May-2016.csv"
#fname <- "E:/EB5205 Clinical/Journal/Appt No Show/Dataset/Kaggle_WY.csv"
#fname_kaggle <- "E:/EB5205 Clinical/Journal/Appt No Show/Dataset/KaggleV2-May-2016.csv"

crs$dataset <- read.csv(
  fname,
  na.strings = c(".", "NA", "", "?"),
  strip.white = TRUE,
  encoding = "UTF-8",
  stringsAsFactors = FALSE
)

crs$dataset_kaggle <- read.csv(
  fname_kaggle,
  na.strings = c(".", "NA", "", "?"),
  strip.white = TRUE,
  encoding = "UTF-8",
  stringsAsFactors = FALSE
)

head(crs$dataset,5)
#===============data prepreprocessing=========================
names(crs$dataset)[names(crs$dataset) == 'Handcap'] <- "Handicap"
names(crs$dataset)[names(crs$dataset) == 'Hipertension'] <- "Hypertension"
names(crs$dataset)[names(crs$dataset) == 'PatientID_Count'] <- "Total_apts"
names(crs$dataset)[names(crs$dataset) == 'Days_To_Appt'] <- "Await_days"
names(crs$dataset)[names(crs$dataset) == 'Book_Day'] <- "Weekday_Book"
names(crs$dataset)[names(crs$dataset) == 'Appt_Day'] <- "Weekday_Appt"

## No_show column: Yes->1, No->0 
crs$dataset$No_show <- ifelse(crs$dataset$No_show=='Yes',1,0)  # works for factors
## Same_Day:Yes:1; No:0.
crs$dataset$Same_Day <- ifelse(crs$dataset$Same_Day=='Yes',1,0)
## Gender:Female:1; Male:0
crs$dataset$Gender <- ifelse(crs$dataset$Gender=='F',1,0)

## Weather: regular:0,Fog:1,Rain:2,Thunderstorm:3.
crs$dataset$Weather[crs$dataset$Weather=='Rain,Thunderstorm'] <- 'Thunderstorm'
crs$dataset$Weather[crs$dataset$Weather=='regular'] <- 0
crs$dataset$Weather[crs$dataset$Weather=='Fog'] <- 1
crs$dataset$Weather[crs$dataset$Weather=='Rain'] <- 2
crs$dataset$Weather[crs$dataset$Weather=='Thunderstorm'] <- 3

## create Missed_Apps variable
Missed_Apps = crs$dataset %>% group_by(PatientId) %>% summarise(Missed_Apps = sum(No_show))
crs$dataset <- dplyr::left_join(crs$dataset,Missed_Apps,by="PatientId")

summary(crs$dataset)
# sapply(crs$dataset,class)
dim(crs$dataset)
str(crs$dataset)
head(crs$dataset,5)

#=============== outliers =========================
#outlier elimination
crs$dataset <- dplyr::filter(crs$dataset,crs$dataset$Age>=0 & crs$dataset$Age<=100)  ## remove 1 row that Age<0
# describe(crs$dataset$Age)
crs$dataset = dplyr::filter(crs$dataset,crs$dataset$Await_days>=0)
dim(crs$dataset)
##missing value##
sapply(crs$dataset, function(x) sum(is.na(x))) ## glad no missing values


#=============== select columns =========================
names(crs$dataset)
droplist <- c("PatientId","AppointmentID","ScheduledDay","AppointmentDay",
              "Neighbourhood","Book_Date","Appt_Date","Weekday_Book",
              "Book_Hr","Age")
droplist1 <- c("PatientId","AppointmentID","ScheduledDay","AppointmentDay",
               "Neighbourhood","Book_Date","Appt_Date","Weekday_Book","Weekday_Appt",
               "Book_Hr","Age","Gender","Alcoholism","Scholarship","Diabetes","Hypertension",
               "Handicap","Weather")
# crs$dataset <- crs$dataset[,-which(names(crs$dataset) %in% droplist)] ## ensemble train/test: 90%,87%
crs$dataset <- crs$dataset[,-which(names(crs$dataset) %in% droplist1)]
names(crs$dataset)

##convert related columns to numerical type before normalization
# if droplist1, then comment below sentence
# crs$dataset$Weather <- sapply(crs$dataset$Weather,as.numeric)
sapply(crs$dataset,class)

##normalization##
no_norm_list = c("Gender","Scholarship","Hypertension","Diabetes","Alcoholism","Same_Day","No_show")
crs$dataset_vars_Norm <- scale(crs$dataset[,-which(names(crs$dataset) %in% no_norm_list)])
crs$dataset <- cbind(crs$dataset_vars_Norm,crs$dataset[,which(names(crs$dataset) %in% no_norm_list)])

head(crs$dataset)

#=============== Spliting training and test dataset================
crs$count.one <- sum(crs$dataset$No_show == '1')
crs$count.zero <- sum(crs$dataset$No_show == '0')
print(crs$count.one)
print(crs$count.zero)
print(c(sum(crs$dataset$No_show=='1'),sum(crs$dataset$No_show=='0')))
print(nrow(crs$dataset))

# Random test data ID
mtx_test <- sampling:::strata(
  crs$dataset,
  stratanames = "No_show",
  size = c(round(crs$count.zero * 0.3), round(crs$count.one * 0.3)),
  method = "srswor"
)

crs$test  <- mtx_test[, 2]

crs$testdata <- crs$dataset[crs$test,]  #testdata: 30% of whole dataset
crs$traindata <- crs$dataset[-crs$test,]  #traindata: 70% of whole dataset

print("size of train and test dataset: ")
print(c(dim(crs$traindata)[1],dim(crs$testdata)[1]))
print("noshows in train and test dataset: ")
print(c(sum(crs$traindata$No_show=='1'),sum(crs$testdata$No_show=='1')))
print("showup in train and test dataset: ")
print(c(sum(crs$traindata$No_show=='0'),sum(crs$testdata$No_show=='0')))


#===========  data balancing by SMOTE ===========
print(nrow(crs$traindata))
print(c(sum(crs$traindata$No_show=="1"),sum(crs$traindata$No_show=="0")))

crs$traindata$No_show <- as.factor(crs$traindata$No_show)
data_smote <- SMOTE(No_show ~., crs$traindata, perc.over=200,k=5)  ##take care,very slooooow,might want save.
crs$traindata <- data_smote

print(nrow(data_smote))
print(c(sum(data_smote$No_show=="1"),sum(data_smote$No_show=="0")))

## after balancing
print("size of train and test dataset after balancing: ")
print(c(dim(crs$traindata)[1],dim(crs$testdata)[1]))
print("noshows in train and test dataset after balancing: ")
print(c(sum(crs$traindata$No_show=='1'),sum(crs$testdata$No_show=='1')))
print("showup in train and test dataset after balancing: ")
print(c(sum(crs$traindata$No_show=='0'),sum(crs$testdata$No_show=='0')))

# print("balanced train dataset No_shows and Showup records: ")
# print(c(sum(crs$traindata$No_show == '1'),sum(crs$traindata$No_show == '0')))

sapply(crs$traindata,class)
sapply(crs$testdata,class)

#=============== modeling ================
crs$dataset.cv_train <- crs$traindata
crs$dataset.cv_test <- crs$testdata

# Change the type of column Class to factor, otherwise it will throw error
crs$dataset.cv_train[, 'No_show'] <-
  as.factor(crs$dataset.cv_train[, 'No_show'])
levels(crs$dataset.cv_train$No_show) <-
  make.names(levels(crs$dataset.cv_train$No_show))

crs$dataset.cv_test[, 'No_show'] <-
  as.factor(crs$dataset.cv_test[, 'No_show'])
levels(crs$dataset.cv_test$No_show) <-
  make.names(levels(crs$dataset.cv_test$No_show))

## show all columns' types
sapply(crs$dataset.cv_train,class)  
sapply(crs$dataset.cv_test,class) 

# define fitControl to decide cv
fitControl <- trainControl(
  method = "cv",
  number = 2,
  # To save out of fold predictions for best parameter combinantions
  savePredictions = 'final',
  classProbs = T # To save the class probabilities of the out of fold predictions
)


#=============== modeling ================
#first layer model: Knn, very slow
knnGrid <-  expand.grid(k = c(2))
crs$model_first_knn <-
  caret::train(
    No_show ~ .,
    data = crs$dataset.cv_train,
    trControl = fitControl,
    method = "knn",
    tuneGrid = knnGrid
  )


# first layer model: Generalized Linear Model
crs$model_first_lr <-
  caret::train (
    No_show ~ .,
    data = crs$dataset.cv_train,
    trControl = fitControl,
    method = "glm"
  )


# first layer model: Neural Network
nnetGrid <-  expand.grid(size = c(2),decay = c(0.0001))
crs$model_first_nnet <-
  caret::train(
    No_show ~ .,
    data = crs$dataset.cv_train,
    trControl = fitControl,
    method = "nnet",
    tuneGrid = nnetGrid
  )

##  Decision Tree
rpart2Grid <-  expand.grid(maxdepth = c(4))
crs$model_first_rpart2 <-
  caret::train(
    No_show ~ .,
    data = crs$dataset.cv_train,
    trControl = fitControl,
    method = "rpart2",
    tuneGrid = rpart2Grid
  )

# Predicting the out of fold prediction probabilities for training data
crs$dataset.cv_train$oof_pred_knn <-
  crs$model_first_knn$pred$X1[order(crs$model_first_knn$pred$rowIndex)]
crs$dataset.cv_train$oof_pred_lr <-
  crs$model_first_lr$pred$X1[order(crs$model_first_lr$pred$rowIndex)]
crs$dataset.cv_train$oof_pred_nnet <-
  crs$model_first_nnet$pred$X1[order(crs$model_first_nnet$pred$rowIndex)]
crs$dataset.cv_train$oof_pred_rpart2 <-
  crs$model_first_rpart2$pred$X1[order(crs$model_first_rpart2$pred$rowIndex)]

# Predicting probabilities for the test data
# crs$dataset.cv_test$oof_pred_knn <-
#   predict(crs$model_first_knn, crs$dataset.cv_test, type = 'prob')$X1  #suddenly doesn't work for droplist1...

crs$dataset.cv_test$oof_pred_lr <-
  predict(crs$model_first_lr, crs$dataset.cv_test, type = 'prob')$X1
crs$dataset.cv_test$oof_pred_nnet <-
  predict(crs$model_first_nnet, crs$dataset.cv_test, type = 'prob')$X1
crs$dataset.cv_test$oof_pred_rpart2 <-
  predict(crs$model_first_rpart2, crs$dataset.cv_test, type = 'prob')$X1


# 2. train second layer model

# Predictors for top layer models
predictors_top <-
  c(
    # 'oof_pred_knn',   #suddenly doesn't work for droplist1...
    'oof_pred_lr',
    'oof_pred_nnet',
    'oof_pred_rpart2'
  )

# second layer model: rpart2
rpart2GridSec <-  expand.grid(maxdepth = c(2))
crs$model_second_rpart2 <-
  train(
    crs$dataset.cv_train[, predictors_top],
    crs$dataset.cv_train[, 'No_show'],
    trControl = fitControl,
    method = "rpart2",
    tuneGrid = rpart2GridSec
  )

# 3.Predict using second layer model
crs$dataset.cv_test$rpart2_stacked <-
  predict(crs$model_second_rpart2, crs$dataset.cv_test[, predictors_top])

# First layer model predicting 0 and 1
# crs$prediction_first_knn <-
#   predict(crs$model_first_knn, newdata = crs$dataset.cv_test)  #suddenly doesn't work for droplist1...
crs$prediction_first_lr <-
  predict(crs$model_first_lr, newdata = crs$dataset.cv_test)
crs$prediction_first_nnet <-
  predict(crs$model_first_nnet, newdata = crs$dataset.cv_test)
crs$prediction_first_rpart2 <-
  predict(crs$model_first_rpart2, newdata = crs$dataset.cv_test)

# Second layer model predicting 0 and 1
crs$prediction_second_rpart2 <-
  predict(crs$model_second_rpart2, newdata = crs$dataset.cv_test)

# 4.ConfusionMatrix output

# First layer model testing data output
# confusionMatrix(crs$prediction_first_knn, crs$dataset.cv_test$No_show, positive = "X1")  #suddenly doesn't work for droplist1...
confusionMatrix(crs$prediction_first_lr, crs$dataset.cv_test$No_show, positive = "X1")
confusionMatrix(crs$prediction_first_nnet, crs$dataset.cv_test$No_show, positive = "X1")
confusionMatrix(crs$prediction_first_rpart2, crs$dataset.cv_test$No_show, positive = "X1")

# Second layer model testing data output
confusionMatrix(crs$prediction_second_rpart2, crs$dataset.cv_test$No_show, positive = "X1")

# First layer model training data output
# confusionMatrix(crs$model_first_knn$pred$obs, crs$model_first_knn$pred$pred, positive = 'X1')  #suddenly doesn't work for droplist1...
confusionMatrix(crs$model_first_lr$pred$obs, crs$model_first_lr$pred$pred, positive = 'X1')
confusionMatrix(crs$model_first_nnet$pred$obs, crs$model_first_nnet$pred$pred, positive = 'X1')
confusionMatrix(crs$model_first_rpart2$pred$obs, crs$model_first_rpart2$pred$pred, positive = 'X1')