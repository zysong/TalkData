library(dplyr)
library(caret)
library(data.table)
library(randomForest)
library(xgboost)
library(mice)
library(MASS)
require(bit64)

pred<-fread("predictors.csv")
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
test_without_events.top20<-gender_age_test %>% left_join(brand_model.top20, by="device_id")
pred.test.without.events.top20<-test_without_events.top20[,-1]

brands_train<-brand_model %>% semi_join(train_with_events, by = "device_id") %>% group_by(phone_brand) %>% 
  summarise(n_devices=n()) %>% arrange(desc(n_devices))
models_train<-brand_model %>% semi_join(train_with_events, by = "device_id") %>% group_by(device_model) %>% 
  summarise(n_devices=n()) %>% arrange(desc(n_devices))
brands_test_only<-n_byBrand %>% semi_join(test_without_events, by="phone_brand") %>% anti_join(train_without_events, by="phone_brand")
models_test_only<-n_byModel %>% semi_join(test_without_events, by="device_model") %>% anti_join(train_without_events, by="device_model")

lm.age<-lm(age~phone_brand+device_model, data=train_without_events.top20)
p.pred.age<-summary(lm.age)$coefficients[,4]
brands.age<-lm.age$xlevels$phone_brand[p.pred.age<0.05]
models.age<-lm.age$xlevels$device_model[p.pred.age<0.05]

lm.age.brand<-lm(age~phone_brand, data=train_without_events.top20)
p.brand.age<-summary(lm.age.brand)$coefficients[,4]
brands.age<-lm.age.brand$xlevels$phone_brand[p.brand.age<0.05]
#age.pred<-predict(lm.age, pred.test.without.events)
lm.age.model<-lm(age~device_model, data=train_without_events.top20)
p.model.age<-summary(lm.age.model)$coefficients[,4]
models.age<-lm.age.model$xlevels$device_model[p.brand<0.05]
glm.gender.brand<-glm(gender~phone_brand, family = "binomial", data=train_without_events.top20)
p.brand.gender<-summary(glm.gender.brand)$coefficients[,4]
brands.gender<-glm.gender.brand$xlevels$phone_brand[p.brand.gender<0.05]
glm.gender.model<-glm(gender~device_model, family = "binomial", data=train_without_events.top20)
p.model.gender<-summary(glm.gender.model)$coefficients[,4]
models.gender<-glm.gender.model$xlevels$device_model[p.model.gender<0.05]

lda.brand<-lda(group~phone_brand, data = train_without_events.top20, CV=TRUE)
lda.brand_model<-lda(group~phone_brand+device_model, data = train_without_events.top20, CV=TRUE)


brand_model.dummy<-dummyVars(~phone_brand, data=rbind(pred.train.without.events, pred.test.without.events))
brand_model.train<-as.data.frame(predict(brand_model.dummy, newdata = pred.train.without.events))
brand_model.test<-predict(brand_model.dummy, newdata = pred.test.without.events)


rf.cv<- rfcv(brand_model.train, class.train.without.events, cv.fold=10)
rf<-randomForest(pred.train, class.train, nree=1000, importance = TRUE)
