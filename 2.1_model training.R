library(tidyverse)
library(caret)
library(skimr)
library(doParallel)
library(caretEnsemble)


load("0_rda/1.1_pre-filter_center_scaled_imputed training and testing dataset_48predictors.rda")
train_imp <- train_imp %>% select(-aki)

##
metric <- "ROC"
trainControl <- trainControl(method="repeatedcv", number=10, repeats=5, search = "grid", savePredictions = "final", 
                             classProbs = TRUE,  summaryFunction = twoClassSummary,
                             verbose = F)
set.seed(202403)
cl <- makeCluster(detectCores())
registerDoParallel(cl)
#------------------------------------
Sys.time()
model_list <- caretList(
  aki2~., data=train_imp,
  trControl=trainControl,
  methodList=c("glm", "glmnet", "rf", "gbm", "xgbTree", 
               "svmLinear", "svmRadial"),
  metric="ROC"
)

save(model_list, file = "0_rda/2.1 model training (7 methods).rda")
Sys.time()

