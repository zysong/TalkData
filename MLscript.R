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

brand_model<-fread("./Data/phone_brand_device_model.csv")
brand_model$device_id<-as.character(brand_model$device_id)
brand_model<-distinct(brand_model[complete.cases(brand_model)])
brand_model<-brand_model[!duplicated(brand_model$device_id),]
brand_model$device_model<-paste(brand_model$phone_brand, brand_model$device_model, sep="-")
#modify pred
#pred<-pred %>% select(-c(phone_brand, device_model)) %>% left_join(brand_model, by="device_id")
#write.csv(pred, "predictors.csv")

n_byBrand<-brand_model %>% group_by(phone_brand) %>% summarise(n_devices=n()) %>% arrange(desc(n_devices))
n_byModel<-brand_model %>% group_by(device_model) %>% summarise(n_devices=n()) %>% arrange(desc(n_devices))
top10brand<-n_byBrand[1:round(nrow(n_byBrand)/10), 1]
top10model<-n_byModel[1:round(nrow(n_byModel)/10), 1]
top20brand<-n_byBrand[1:round(nrow(n_byBrand)/5), 1]
top20model<-n_byModel[1:round(nrow(n_byModel)/5), 1]
brand_model.top10<-brand_model.top20<-brand_model
brand_model.top10$phone_brand[!(brand_model$phone_brand %in% top10brand$phone_brand)]<-"minor_brand"
brand_model.top10$device_model[!(brand_model$device_model %in% top10model$device_model)]<-"minor_model"
brand_model.top20$phone_brand[!(brand_model$phone_brand %in% top20brand$phone_brand)]<-"minor_brand"
brand_model.top20$device_model[!(brand_model$device_model %in% top20model$device_model)]<-"minor_model"

train_with_events<-inner_join(gender_age_train, pred, by="device_id")
pred.train<-train_with_events[,-(1:5)]
train_without_events<-inner_join(gender_age_train, brand_model, by="device_id")
pred.train.without.events<-train_without_events[,-(1:4)]
train_without_events.top20<-inner_join(gender_age_train, brand_model.top20, by="device_id")
pred.train.without.events.top20<-train_without_events.top20[,-(1:4)]
test_with_events<-inner_join(gender_age_test, pred, by="device_id")
pred.test<-test_with_events[,-(1:2)]
test_without_events<-gender_age_test %>% left_join(brand_model, by="device_id")
pred.test.without.events<-test_without_events[,-1]

train()
lm.age.brand.top20<-lm(age~phone_brand, data=train_without_events.top20)
summary(lm.age.brand.top20)
lm.age<-lm(age~phone_brand, data=train_without_events)
step(lm.age)
summary.lm.age<-summary(lm.age)
age.pred<-predict(lm.age, pred.test.without.events)

brand_model.dummy<-dummyVars(~phone_brand + device_model, data=rbind(pred.train.without.events, pred.test.without.events))
brand_model.train<-predict(brand_model.dummy, newdata = pred.train.without.events)
brand_model.test<-predict(brand_model.dummy, newdata = pred.test.without.events)

rf.cv<- rfcv(brand_model.train, class.train.without.events, cv.fold=10)
rf<-randomForest(pred.train, class.train, nree=1000, importance = TRUE)
