library(data.table)
library(dplyr)
library(stringr)
library(ggplot2)
library(caret)
library(useful)
library(mice)
require(bit64)

events<-fread("events.csv")
events$device_id<-as.character(events$device_id)
events$event_id<-as.character(events$event_id)
#head(events)
#derive time info from timestamp
#library(chron)
#date_time<-str_split_fixed(events$timestamp, " ", 2)
#events$time<-chron(times. = date_time[,2], format = "h:m:s")
#events$date<-chron(dates. = date_time[,1], format = "y-m-d")
#events_byDate<-group_by(events, date) %>% summarise(n())
events$hour<-as.POSIXlt(events$timestamp)$hour
#events_byHour<-group_by(events, hour) %>% summarise(n())
events<-select(events, -timestamp)
#identify missing location data
events$longitude[events$longitude==0.00]<-NA
events$latitude[events$latitude==0.00]<-NA
#select the complete dataset (about half includes missing location data!)
events.com<-events[complete.cases(events),]
#identify the primary location and the maximal travel distance
events_byDeviceLocation<-events.com %>% group_by(device_id, longitude, latitude) %>% 
  summarise(n_events=n())
device_loc<-events_byDeviceLocation %>% group_by(device_id) %>% 
  summarise(longitude1=longitude[which.max(n_events)], latitude1=latitude[which.max(n_events)], 
            max.dist=max(as.matrix(dist(cbind(longitude, latitude), diag=TRUE))))

events_byDevice<-events %>% group_by(device_id) %>% summarise(n_events=n())
#head(events_byDevice)
#hist(events_byDevice$n_events, xlim = c(0, 1000), breaks = 5000)
#events_byDate<-events %>% group_by(date) %>% summarise(n_events=n())
events_byDeviceHour<-events %>% group_by(device_id, hour) %>% summarise(n_events=n())
device_hours<-events_byDeviceHour %>% group_by(device_id) %>% arrange(desc(n_events)) %>% 
  summarise(hour1=hour[1], hour2=hour[2], distinct_hours=n_distinct(hour))
remove(events.com)

#plot the temporal pattern of events per device for each group of users

#gender_age_events<-inner_join(x=train_fold1, y=events, by = "device_id")
#events_byGroupHour<-gender_age_events %>% group_by(hour, group) %>% 
#  summarise(events_pd=n()/length(unique(device_id)))
#ggplot(events_byGroupHour, aes(x = hour, y = events_pd, colour = group)) +
#  geom_point()

#join devices and labels
app_events<-fread("app_events.csv")
app_events$event_id<-as.character(app_events$event_id)
app_events$app_id<-as.character(app_events$app_id)
app_label<-fread("app_labels.csv")
app_label$app_id<-as.character(app_label$app_id)
#head(app_events, 10)
device_apps<-events %>% select(c(device_id, event_id)) %>% 
  inner_join(app_events, by = "event_id")
remove(app_events)
#count the total number of installed/active apps per device
device_apps_installed <-device_apps %>% group_by(device_id) %>% 
  summarise(n_app_installed= n_distinct(app_id))
device_apps_active <-device_apps %>% filter(is_active==1) %>% group_by(device_id) %>% 
  summarise(n_app_active= n_distinct(app_id))
#installed apps
device_labels_installed <- device_apps %>% select(device_id, app_id) %>% distinct() %>% 
  inner_join(app_label, by="app_id")
installation_byDeviceLabel <- device_labels_installed %>% group_by(device_id, label_id) %>% 
  summarise(n_app=n())
#device_labels_top_installed<- installation_byDeviceLabel %>% group_by(device_id) %>% 
#  arrange(desc(n_app)) %>% summarise(label1=label_id[1], label2=label_id[2])
label_installed_wide<-dcast(installation_byDeviceLabel, as.factor(device_id)~as.factor(label_id), 
                            value.var="n_app", fill=0)
names(label_installed_wide)[1]<-'device_id'
remove(device_labels_installed, installation_byDeviceLabel)
#active apps
device_labels_active <- device_apps %>% select(device_id, app_id, is_active) %>% 
  inner_join(app_label, by="app_id")
active_byDeviceLabel <- device_labels_active %>% group_by(device_id, label_id) %>% 
  summarise(n_app=sum(is_active))
#device_labels_top_active<- active_byDeviceLabel %>% group_by(device_id) %>% 
#  arrange(desc(n_app)) %>% summarise(label1=label_id[1])
label_active_wide<-dcast(active_byDeviceLabel, as.factor(device_id)~as.factor(label_id), 
                         value.var="n_app", fill=0)
names(label_active_wide)[1]<-'device_id'
remove(device_labels_active, active_byDeviceLabel)

#clean memory
remove(device_apps)
#principal component analysis
pca.label.installed<-prcomp(label_installed_wide[,-1], scale. = TRUE, center = TRUE)
#biplot(pca.label.installed)
#screeplot(pca.label.installed)
#to scale apps without active use, standard deviation is set to 1
sd.active<-apply(label_active_wide[,-1], 2, sd)
sd.active[sd.active==0]<-1
pca.label.active<-prcomp(label_active_wide[,-1], scale. = sd.active, center = TRUE)
#screeplot(pca.label.active)
device_label_pc<-as.data.frame(cbind(pca.label.installed$x[,1:5], pca.label.active$x[,1:5]))
names(device_label_pc)<- c(paste0(rep("PC", 5), 1:5, ".ins"), paste0(rep("PC", 5), 1:5, ".act"))
device_label_pc$device_id<-as.integer64(as.character(label_active_wide$device_id))

#find the most common category
#label_cat<-fread("label_categories.csv")
#label_cat$category[events_byLabel$label_id[which.max(events_byLabel$count)]]

#gender_age_events_apps<-inner_join(gender_age_events, app_events, by="event_id")
#group_events_app_label<-inner_join(gender_age_events_apps, app_label, by="app_id")
#events_byLabel<-group_events_app_label %>% group_by(label_id) %>% summarise(count=n())
#ggplot(events_byLabel, aes(x = label_id, y = count)) +
#  geom_point()

gender_age_train<-fread("gender_age_train.csv")
gender_age_test<-fread("gender_age_test.csv")
sample_sub<-fread("sample_submission.csv")

brand_model<-fread("phone_brand_device_model.csv")
brand_model$device_id<-as.character(brand_model$device_id)
brand_model<-distinct(brand_model[complete.cases(brand_model)])

train_with_events<-semi_join(gender_age_train, events_byDevice, by="device_id")
train_without_events<-anti_join(gender_age_train, events_byDevice, by="device_id")
test_with_events<-semi_join(gender_age_test, events_byDevice, by="device_id")
test_without_events<-anti_join(gender_age_test, events_byDevice, by="device_id")

pred<-list(events_byDevice, device_apps_installed, device_apps_active, 
           device_label_pc, device_hours, device_loc, brand_model) %>%
  Reduce(function(dtf1,dtf2) left_join(dtf1,dtf2,by="device_id"), .)

md.pattern(pred)

sum(is.na(events_byDevice$device_id))
