library(dplyr)
library(caret)
library(data.table)
library(randomForest)
library(xgboost)
library(mice)
require(bit64)

pred<-fread("predictors.csv", stringsAsFactors = TRUE)
pred$device_id<-as.character(pred$device_id)

md.pattern(pred)

gender_age_train<-fread("./Data/gender_age_train.csv", stringsAsFactors = TRUE)
gender_age_train$device_id<-as.character(gender_age_train$device_id)
gender_age_test<-fread("./Data/gender_age_test.csv")
gender_age_test$device_id<-as.character(gender_age_test$device_id)
sample_sub<-fread("./Data/sample_submission.csv")

brand_model<-fread("./Data/phone_brand_device_model.csv", stringsAsFactors = TRUE)
brand_model$device_id<-as.character(brand_model$device_id)
brand_model<-distinct(brand_model[complete.cases(brand_model)])
brand_model<-brand_model[!duplicated(brand_model$device_id),]

train_with_events<-inner_join(gender_age_train, pred, by="device_id")
pred.train<-train_with_events[,-(1:5)]
class.train<-train_with_events$group
train_without_events<-inner_join(gender_age_train, brand_model, by="device_id")
pred.train.without.events<-train_without_events[,-(1:4)]
class.train.without.events<-train_without_events$group
test_with_events<-inner_join(gender_age_test, pred, by="device_id")
pred.test<-test_with_events[,-(1:2)]
test_without_events<-gender_age_test %>% left_join(brand_model, by="device_id")
pred.test.without.events<-test_without_events[,-1]

brand_model.dummy<-dummyVars(~phone_brand + device_model, data=rbind(pred.train.without.events, pred.test.without.events))
brand_model.train<-predict(brand_model.dummy, newdata = pred.train.without.events)
brand_model.test<-predict(brand_model.dummy, newdata = pred.test.without.events)

rf.cv<- rfcv(brand_model.train, class.train.without.events, cv.fold=10)
rf<-randomForest(pred.train, class.train, nree=1000, importance = TRUE)
