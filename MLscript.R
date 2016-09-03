require(plyr)
library(dplyr)
library(tidyr)
library(caret)
library(data.table)
library(randomForest)
library(xgboost)
library(mice)
library(MASS)
require(bit64)
require(e1071)
require(doMC)
require(pROC)

pred<-fread("predictors.csv")
pred$tot.hours<-rowSums(dplyr::select(pred, h0:h23))

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
train_with_events<-gender_age_train[pred, nomatch=0L, on="device_id"]
train_with_events$group<-make.names(train_with_events$group)
pred.train<-train_with_events[,-(1:5),with=FALSE]
test_with_events<-gender_age_test[pred, nomatch=0L, on="device_id"]
pred.test<-test_with_events[,-(1:2),with=FALSE]
train_without_events<-gender_age_train[brand_model, nomatch=0L, on="device_id"]
train_without_events$group<-make.names(train_without_events$group)
pred.train.without.events<-train_without_events[,-(1:4),with=FALSE]

#find the optimal truncation
ldaResults<-data.frame()
for (percent in (1:5)/10) {
  ##truncate the brand and model lists
  truncBrand<-n_byBrand[1:round(nrow(n_byBrand)*percent), 1]
  truncModel<-n_byModel[1:round(nrow(n_byModel)*percent), 1]
  brand_model.trunc<-brand_model
  brand_model.trunc$phone_brand[!(brand_model$phone_brand %in% truncBrand$phone_brand)]<-"minor_brand"
  brand_model.trunc$device_model[!(brand_model$device_model %in% truncModel$device_model)]<-"minor_model"
  train_without_events.trunc<-inner_join(gender_age_train, brand_model.trunc, by="device_id")
  pred.train.without.events.trunc<-train_without_events.trunc[,-(1:4)]
  brand_model.dummy<-dummyVars(~phone_brand+device_model, data=rbind(pred.train.without.events.trunc, pred.test.without.events.trunc))
  brand_model.train<-as.data.frame(predict(brand_model.dummy, newdata = pred.train.without.events.trunc))
  brand_model.train<-dplyr::select(brand_model.train, -c(phone_brandminor_brand, device_modelminor_model))
  #register paralell computing
  library(doMC)
  registerDoMC(cores = 2)
  #train a lda model
  ldaCtrl<-trainControl(method = "cv", number = 10,
                        summaryFunction = multiClassSummary, classProbs = TRUE)
  ldaGrid<-expand.grid(dimen=1:5)
  set.seed(101)
  ldaFit<-train(x = brand_model.train, 
                y = train_without_events$group,
                method = "lda2",
                preProcess = c("center", "scale"),
                metric = "logLoss",
                tuneGrid = ldaGrid,
                trControl = ldaCtrl)
  ldaResults<-rbind(ldaResults, cbind(ldaFit$results[,1:2], trunc=rep(percent, 5)))
}
#find the optimal parameters
ldaResults.wide<-spread(ldaResults, trunc, logLoss)
matlines(ldaResults.wide[,1], ldaResults.wide[,-1])
percent<-.1
dimen<-3
#run the lda model again with the chosen parameters
truncBrand<-n_byBrand[1:round(nrow(n_byBrand)*percent), 1]
truncModel<-n_byModel[1:round(nrow(n_byModel)*percent), 1]
brand_model.trunc<-brand_model
brand_model.trunc$phone_brand[!(brand_model$phone_brand %in% truncBrand$phone_brand)]<-"minor_brand"
brand_model.trunc$device_model[!(brand_model$device_model %in% truncModel$device_model)]<-"minor_model"
train_without_events.trunc<-inner_join(gender_age_train, brand_model.trunc, by="device_id")
pred.train.without.events.trunc<-train_without_events.trunc[,-(1:4)]
test_without_events.trunc<-gender_age_test %>% left_join(brand_model.trunc, by="device_id")
pred.test.without.events.trunc<-test_without_events.trunc[,-1]
brand_model.dummy<-dummyVars(~phone_brand+device_model, data=rbind(pred.train.without.events.trunc, pred.test.without.events.trunc))
brand_model.train<-as.data.frame(predict(brand_model.dummy, newdata = pred.train.without.events.trunc))
brand_model.test<-as.data.frame(predict(brand_model.dummy, newdata = pred.test.without.events.trunc))
brand_model.train<-dplyr::select(brand_model.train, -c(phone_brandminor_brand, device_modelminor_model))
brand_model.test<-dplyr::select(brand_model.test, -c(phone_brandminor_brand, device_modelminor_model))
#train a lda model
ldaCtrl<-trainControl(method = "cv", number = 10,
                      summaryFunction = multiClassSummary, classProbs = TRUE)
ldaGrid<-expand.grid(dimen=dimen)
set.seed(101)
ldaFit<-train(x = brand_model.train, 
              y = train_without_events$group,
              method = "lda2",
              preProcess = c("center", "scale"),
              metric = "logLoss",
              tuneGrid = ldaGrid,
              trControl = ldaCtrl)

testProbs<-predict(ldaFit, newdata=brand_model.test, type = "prob")
testProbs0<-cbind(test_without_events.trunc$device_id, testProbs)
write.csv(testProbs0, "group_test_no_events.csv")

#train an xgboost model
#xgbGrid0 <- expand.grid(nrounds = seq(20, 100, by=20),
#                       eta = .1, 
#                       max_depth = c(2,4,6),
#                       gamma = .1,
#                       colsample_bytree = 1,
#                       min_child_weight = 1)

#xgbCtrl0<-trainControl(method = "cv", number = 10,
#                      summaryFunction = multiClassSummary, classProbs = TRUE)

#xgbTune0 <- train(brand_model.train, train_without_events$group, 
#                 method = "xgbTree", 
#                 preProcess = c("center", "scale"),
#                 tuneGrid = xgbGrid0,
#                 metric = "logLoss",
#                 trControl = xgbCtrl0)

#plot(xgbTune0, auto.key= list(columns = 2, lines = TRUE))

brands_train<-brand_model %>% semi_join(train_with_events, by = "device_id") %>% group_by(phone_brand) %>% 
  summarise(n_devices=n()) %>% arrange(desc(n_devices))
models_train<-brand_model %>% semi_join(train_with_events, by = "device_id") %>% group_by(device_model) %>% 
  summarise(n_devices=n()) %>% arrange(desc(n_devices))
brands_test_only<-n_byBrand %>% semi_join(test_without_events, by="phone_brand") %>% anti_join(train_without_events, by="phone_brand")
models_test_only<-n_byModel %>% semi_join(test_without_events, by="device_model") %>% anti_join(train_without_events, by="device_model")


lm.age<-lm(train_without_events$age~., data=brand_model.train)
p.pred.age<-summary(lm.age)$coefficients[,4]
brand_model.age<-names(lm.age$model)[p.pred.age<0.01][-1]

glm.gender<-glm(train_without_events$gender~., family = "binomial", data=brand_model.train)
p.pred.gender<-summary(glm.gender)$coefficients[,4]
brand_model.gender<-names(glm.gender$model)[p.pred.gender<0.01][-1]

#Keep only the significant brands and models
brand_model.selected<-unique(c(names(brand_model.train), names(brand_model.test)))
brand_model.selected<-gsub("phone_brand", "", brand_model.selected) 
brand_model.selected<-gsub("device_model", "", brand_model.selected)
pred.train$phone_brand<-as.character(pred.train$phone_brand)
pred.train$device_model<-as.character(pred.train$device_model)
pred.test$phone_brand<-as.character(pred.test$phone_brand)
pred.test$device_model<-as.character(pred.test$device_model)
pred.train$phone_brand[!(pred.train$phone_brand %in% brand_model.selected)]<-"other"
pred.train$device_model[!(pred.train$device_model %in% brand_model.selected)]<-"other"
pred.test$phone_brand[!(pred.test$phone_brand %in% brand_model.selected)]<-"other"
pred.test$device_model[!(pred.test$device_model %in% brand_model.selected)]<-"other"

pred.dummy<-dummyVars(~phone_brand+device_model, data=rbind(pred.train, pred.test))
pred.train.dummies<-as.data.frame(predict(pred.dummy, newdata=pred.train))
pred.test.dummies<-as.data.frame(predict(pred.dummy, newdata=pred.test))
pred.train.wide<-cbind(pred.train[,c("phone_brand", "device_model"):=NULL], pred.train.dummies)
pred.test.wide<-cbind(pred.test[,c("phone_brand", "device_model"):=NULL], pred.test.dummies)

#train an xgboost model
library(doMC)
registerDoMC(cores = 2)
set.seed(101)
xgbGrid <- expand.grid(nrounds = seq(20, 100, by=20),
                       eta = .1, 
                       max_depth = c(3, 4, 5),
                       gamma = .1,
                       colsample_bytree = 1,
                       min_child_weight = 1)

xgbCtrl<-trainControl(method = "cv", number = 10,
                   summaryFunction = multiClassSummary, classProbs = TRUE)

xgbTune <- train(pred.train.wide, train_with_events$group, 
                 method = "xgbTree", 
                 preProcess = c("center", "scale"),
                 tuneGrid = xgbGrid,
                 metric = "logLoss",
                 trControl = xgbCtrl)

plot(xgbTune, auto.key= list(columns = 2, lines = TRUE))
varImp(xgbTune)
xgbTune$bestTune

sample_sub<-fread("./Data/sample_submission.csv")
group.test.withEvents<-predict(xgbTune$finalModel, newdata=as.matrix(pred.test.wide)) %>% matrix(nrow=nrow(pred.test), ncol=12, byrow=TRUE)
group.test.withEvents<-as.data.frame(group.test.withEvents)
testProbs.withEvents<-cbind(test_with_events$device_id, group.test.withEvents)
names(testProbs0)<-names(sample_sub)
names(testProbs.withEvents)<-names(sample_sub)
testProbs.noEvents<-anti_join(testProbs0, testProbs.withEvents, by="device_id")
device_id_test<-data.frame(device_id=as.factor(sample_sub$device_id))
df_sub<-rbind(testProbs.noEvents, testProbs.withEvents) %>% right_join(device_id_test, by="device_id")
write.csv(df_sub, "submission.csv", row.names = FALSE)
