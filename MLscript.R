library(dplyr)
library(caret)
library(data.table)
library(randomForest)
library(xgboost)
require(bit64)

pred<-fread("predictors.csv", stringsAsFactors = TRUE)
pred$device_id<-as.character(pred$device_id)
gender_age_train<-fread("./Data/gender_age_train.csv", stringsAsFactors = TRUE)
gender_age_train$device_id<-as.character(gender_age_train$device_id)
gender_age_test<-fread("./Data/gender_age_test.csv")
gender_age_test$device_id<-as.character(gender_age_test$device_id)
sample_sub<-fread("sample_submission.csv")

train_with_events<-inner_join(gender_age_train, pred, by="device_id")
pred.train<-train_with_events[,-(1:5)]
class.train<-train_with_events$group
train_without_events<-inner_join(gender_age_train, select(pred, c(device_id, phone_brand, device_model)), by="device_id")
test_with_events<-semi_join(gender_age_test, pred, by="device_id")
test_without_events<-anti_join(gender_age_test, pred, by="device_id")

rf.cv<- rfcv(pred.train, class.train, cv.fold=10)
rf<-randomForest(pred.train, class.train, nree=1000, importance = TRUE)
