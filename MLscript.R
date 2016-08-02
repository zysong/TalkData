library(dplyr)
library(caret)
library(data.table)
library(randomForest)
library(xgboost)
library(mice)
require(bit64)

pred<-fread("predictors.csv", stringsAsFactors = TRUE)
pred$device_id<-as.character(pred$device_id)
#temporary modification
pred<-pred %>% select(-c(hour1, hour2, distinct_hours)) %>% left_join(device_hours, by = "device_id")

md.pattern(pred)

gender_age_train<-fread("./Data/gender_age_train.csv", stringsAsFactors = TRUE)
gender_age_train$device_id<-as.character(gender_age_train$device_id)
gender_age_test<-fread("./Data/gender_age_test.csv")
gender_age_test$device_id<-as.character(gender_age_test$device_id)
sample_sub<-fread("./Data/sample_submission.csv")

train_with_events<-inner_join(gender_age_train, pred, by="device_id")
pred.train<-train_with_events[,-(1:5)]
class.train<-train_with_events$group
train_without_events<-inner_join(gender_age_train, select(pred, c(device_id, phone_brand, device_model)), by="device_id")
test_with_events<-semi_join(gender_age_test, pred, by="device_id")
test_without_events<-anti_join(gender_age_test, pred, by="device_id")

brand_model<-fread("./Data/phone_brand_device_model.csv")
brand_model$device_id<-as.character(brand_model$device_id)
brand_model<-distinct(brand_model[complete.cases(brand_model)])
brand_model<-brand_model[!duplicated(brand_model$device_id),]
brand_model.train<- semi_join(brand_model, gender_age_train, by="device_id")
brand_model.test<- semi_join(brand_model, test_without_events, by="device_id")
brand_model.dummy<-dummyVars(~phone_brand + device_model, data=rbind(brand_model.train, brand_model.test))
brand_model.train<-predict(brand_model.dummy, newdata = brand_model.train)
brand_model.test<-predict(brand_model.dummy, newdata = brand_model.test)

rf.cv<- rfcv(pred.train, class.train, cv.fold=10)
rf<-randomForest(pred.train, class.train, nree=1000, importance = TRUE)
