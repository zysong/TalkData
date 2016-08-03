library(dplyr)
library(caret)
library(data.table)
library(randomForest)
library(xgboost)
library(mice)
require(bit64)

pred<-fread("predictors.csv", stringsAsFactors = TRUE)
pred$device_id<-as.character(pred$device_id)

gender_age_train<-fread("./Data/gender_age_train.csv", stringsAsFactors = TRUE)
gender_age_train$device_id<-as.character(gender_age_train$device_id)
gender_age_test<-fread("./Data/gender_age_test.csv")
gender_age_test$device_id<-as.character(gender_age_test$device_id)
sample_sub<-fread("./Data/sample_submission.csv")

brand_model<-fread("./Data/phone_brand_device_model.csv", stringsAsFactors = TRUE)
brand_model$device_id<-as.character(brand_model$device_id)
brand_model<-distinct(brand_model[complete.cases(brand_model)])
brand_model<-brand_model[!duplicated(brand_model$device_id),]
brand_model$device_model<-paste(brand_model$phone_brand, brand_model$device_model, sep="-")
n_byBrand<-brand_model %>% group_by(phone_brand) %>% summarise(n_devices=n()) %>% arrange(desc(n_devices))
n_byModel<-brand_model %>% group_by(device_model) %>% summarise(n_devices=n()) %>% arrange(desc(n_devices))
topBrands<-subset(n_byBrand, n_devices>5000)$phone_brand
topModels<-subset(n_byModel, n_devices>1000)$device_model

train_with_events<-inner_join(gender_age_train, pred, by="device_id")
pred.train<-train_with_events[,-(1:5)]
train_without_events<-inner_join(gender_age_train, brand_model, by="device_id")
pred.train.without.events<-train_without_events[,-(1:4)]
test_with_events<-inner_join(gender_age_test, pred, by="device_id")
pred.test<-test_with_events[,-(1:2)]
test_without_events<-gender_age_test %>% left_join(brand_model, by="device_id")
pred.test.without.events<-test_without_events[,-1]

lm.age<-lm(train_without_events$age~., data=pred.train.without.events)
summary.lm.age<-summary(lm.age)
age.pred<-predict(lm.age, pred.test.without.events)

brand_model.dummy<-dummyVars(~phone_brand + device_model, data=rbind(pred.train.without.events, pred.test.without.events))
brand_model.train<-predict(brand_model.dummy, newdata = pred.train.without.events)
brand_model.test<-predict(brand_model.dummy, newdata = pred.test.without.events)

rf.cv<- rfcv(brand_model.train, class.train.without.events, cv.fold=10)
rf<-randomForest(pred.train, class.train, nree=1000, importance = TRUE)
