save.image('F:\\2020\\study\\01_축산도체\\1.1_등급 예측 구현(일반)\\1_R분석\\210203_(전체농가_등외제외)1차등급예측2.RData')
load('F:\\2020\\study\\01_축산도체\\1.1_등급 예측 구현(일반)\\1_R분석\\210203_(전체농가_등외제외)1차등급예측2.RData')

#install.packages("caret")
#install.packages("e1071")
#install.packages("randomForest")
#install.packages("scales")
#install.packages("dplyr")
#install.packages("gbm")
#install.packages("readr")
#install.packages("readxl")

library(MASS)
library(nnet)
library(caret)
library(e1071)
library(rpart)
library(randomForest)
library(scales)
library(class)
library(dplyr)
library(gbm)
library(readr)
library(readxl)

dataset <- read_excel(path = "F:\\2020\\study\\01_축산도체\\0_dataset\\0_데이터매칭\\농가전체(등급).xlsx", col_names= TRUE)
dataset <- dataset[,c("도체중", "등지방두께", "1차등급")]

dataset=dataset[-which(dataset$`1차등급`=="등외"),]
dataset$`1차등급` <- as.factor(as.numeric(factor(dataset$`1차등급`, level=c("1+","1","2"))))

set.seed(2021)
train_idx1 <- sample(which(dataset$`1차등급`==1), size=0.8*sum(dataset$`1차등급`==1), replace=F) # train-set 0.8, test-set 0.2
train_idx2 <- sample(which(dataset$`1차등급`==2), size=0.8*sum(dataset$`1차등급`==2), replace=F) # train-set 0.8, test-set 0.2
train_idx3 <- sample(which(dataset$`1차등급`==3), size=0.8*sum(dataset$`1차등급`==3), replace=F) # train-set 0.8, test-set 0.2
train_idx <- c(train_idx1,train_idx2,train_idx3)
test_idx <- c(1:nrow(dataset))[-train_idx]

train <- dataset[train_idx, ]
test <- dataset[test_idx, ]


#다항 로지스틱
model2 = multinom(`1차등급`~., data=train)
summary(model2)
model_pr1 <- predict(model2, train)
confusionMatrix(model_pr1,train$`1차등급`)
model_pr<-predict(model2, test)
confusionMatrix(model_pr,test$`1차등급`)


#http://www.dodomira.com/2016/05/29/564/ 

rpartmod <- rpart(`1차등급`~. , data=train, method="class")
plot(rpartmod)
text(rpartmod)
printcp(rpartmod)
plotcp(rpartmod)

ptree <- prune(rpartmod, cp= rpartmod$cptable[which.min(rpartmod$cptable[,"xerror"]),"CP"])
plot(ptree)
text(ptree)

rpartpred1<-predict(ptree, train, type='class')
confusionMatrix(rpartpred1, train$`1차등급`)

rpartpred<-predict(ptree, test, type='class')
confusionMatrix(rpartpred, test$`1차등급`)


#https://data-make.tistory.com/81
train <- droplevels(train)

forest_m <- randomForest(`1차등급` ~ ., data=train)
forest_m   

prd_v1 <- predict(forest_m, newdata = train, type = 'class')
confusionMatrix(prd_v1, train$`1차등급`)

prd_v <- predict(forest_m, newdata = test, type = 'class')
confusionMatrix(prd_v, test$`1차등급`)


#knn
train_x <- train[, -3] 
test_x <- test[, -3] 

train_y <- train[, 3] 
test_y <- test[, 3]

train_yy=as.vector(as.matrix(train_y))
test_yy=as.vector(as.matrix(test_y))

knn_1 <- knn(train = train_x, test = train_x, cl = train_yy, k = 1)
#knn_1 결과: train_x의 예측 y값
accuracy_1 <- sum(knn_1 == train_yy) / length(train_yy)
accuracy_1
#train 정확도
confusionMatrix(knn_1,as.factor(train_yy))

knn_1 <- knn(train = train_x, test = test_x, cl = train_yy, k = 1)
#knn_1 결과: test_x의 예측 y값
accuracy_1 <- sum(knn_1 == test_yy) / length(test_yy)
accuracy_1
#test 정확도
confusionMatrix(knn_1,as.factor(test_yy))


#SVM
sv <- svm(`1차등급` ~., data =train,type = "C-classification")
summary(sv)

svp1<-predict(sv, train)
confusionMatrix(svp1,train$`1차등급`)

svp<-predict(sv, test)
confusionMatrix(svp,test$`1차등급`)


#Navie Bayes Classifier
nb_model <- naiveBayes(`1차등급`~.,data = train)
nb_model

nbpred1 <- predict(nb_model, train, type='class')
confusionMatrix(nbpred1, train$`1차등급`)

nbpred <- predict(nb_model, test, type='class')
confusionMatrix(nbpred, test$`1차등급`)


#Gradient Boosting Classifier https://www.datatechnotes.com/2018/03/classification-with-gradient-boosting.html
mod_gbm=gbm(`1차등급`~., data=train, cv.folds=10,shrinkage=.01, n.minobsinnode = 10, n.trees = 200)

pred2 = predict(object = mod_gbm, newdata = train, n.trees = 200, type = "response")
labels2 = colnames(pred2)[apply(pred2, 1, which.max)]
confusionMatrix(train$`1차등급`, as.factor(labels2))

pred = predict(object = mod_gbm, newdata = test, n.trees = 200, type = "response")
labels = colnames(pred)[apply(pred, 1, which.max)]
confusionMatrix(factor(test$`1차등급`), as.factor(labels))
