library(dplyr)
library(caret)
library(data.table)
library(randomForest)
library(xgboost)
library(mice)
library(MASS)
require(bit64)
require(e1071)

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

#models_byBrand_train <- train_without_events.top20 %>% group_by(phone_brand) %>% summarise(n_model=n_distinct(device_model))

brands_train<-brand_model %>% semi_join(train_with_events, by = "device_id") %>% group_by(phone_brand) %>% 
  summarise(n_devices=n()) %>% arrange(desc(n_devices))
models_train<-brand_model %>% semi_join(train_with_events, by = "device_id") %>% group_by(device_model) %>% 
  summarise(n_devices=n()) %>% arrange(desc(n_devices))
brands_test_only<-n_byBrand %>% semi_join(test_without_events, by="phone_brand") %>% anti_join(train_without_events, by="phone_brand")
models_test_only<-n_byModel %>% semi_join(test_without_events, by="device_model") %>% anti_join(train_without_events, by="device_model")

brand_model.dummy<-dummyVars(~phone_brand+device_model, data=rbind(pred.train.without.events.top20, pred.test.without.events.top20))
brand_model.train<-as.data.frame(predict(brand_model.dummy, newdata = pred.train.without.events.top20))
brand_model.test<-as.data.frame(predict(brand_model.dummy, newdata = pred.test.without.events.top20))
brand_model.train<-dplyr::select(brand_model.train, -c(phone_brandminor_brand, device_modelminor_model))
brand_model.test<-dplyr::select(brand_model.test, -c(phone_brandminor_brand, device_modelminor_model))
brand.dummy<-dummyVars(~phone_brand, data=rbind(pred.train.without.events.top20, pred.test.without.events.top20))
brand.train<-as.data.frame(predict(brand.dummy, pred.train.without.events.top20))
brand.train<-dplyr::select(brand.train, -phone_brandminor_brand)
#brand_model.train.preProcess<-preProcess(brand_model.train)
#brand_model.train.scaled<-predict(brand_model.train.preProcess, brand_model.train)
#brand_model.test.scaled<-predict(brand_model.train.preProcess, brand_model.test)

ctrl<-trainControl(method = "repeatedcv", number = 10, repeats = 3,
                   summaryFunction = multiClassSummary, classProbs = TRUE)
set.seed(101)
ldaFit<-train(x = brand_model.train, 
              y = make.names(train_without_events$group),
              method = "lda2",
              preProcess = c("center", "scale"),
              metric = "logLoss",
              tuneLength = 10,
              trControl = ctrl)

testProbs<-predict(ldaFit, newdata=brand_model.test, type = "prob")

lm.age<-lm(train_without_events$age~., data=brand_model.train)
p.pred.age<-summary(lm.age)$coefficients[,4]
brand_model.age<-names(lm.age$model)[p.pred.age<0.01][-1]

glm.gender<-glm(train_without_events$gender~., family = "binomial", data=brand_model.train)
p.pred.gender<-summary(glm.gender)$coefficients[,4]
brand_model.gender<-names(glm.gender$model)[p.pred.gender<0.01][-1]

brand_model.selected<-unique(c(brand_model.age, brand_model.gender))

lda.brand<-lda(group~phone_brand, data = train_without_events.top20, CV=TRUE)
lda.brand_model<-lda(group~phone_brand+device_model, data = train_without_events.top20, CV=TRUE)





rf.cv<- rfcv(brand_model.train, class.train.without.events, cv.fold=10)
rf<-randomForest(pred.train, class.train, nree=1000, importance = TRUE)
