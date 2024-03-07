library(tidyverse)
library(caret)
library(caretEnsemble)
library(gbm)
library(pROC)
library(glue)

load("0_rda/1.1_pre-filter_center_scaled_imputed training and testing dataset_48predictors.rda")
# test_imp 

load("0_rda/2.1 model training (7 methods).rda")
modelnames <- c("Classic logistic regression", 
                "Penalized logistic regression", 
                "Random forest", 
                "Stochastic gradient boosting", 
                "Extreme gradient boosting", 
                "Support vector machines (linear kernel)", 
                "Support vector machines (Gaussian kernel)")

names(model_list) <- modelnames

#---------------------------------------------
# performance of final models by algorithms in the training data
model_list %>% resamples %>% summary
model_list %>% resamples %>% dotplot(main = "Model performance by each algorithm for the training dataset")

#---------------------------------------------
# Feature importance
get.imp <- function(i) {
  tpd <-  varImp(model_list[[i]])$importance %>%
    data.frame()
  if(dim(tpd)[2] == 2) {tpd <-  select(tpd, "yes")}
  names(tpd) <- names(model_list[i]) 
  tpd %>% rownames_to_column()
}
imp_by_models <- get.imp(1) %>% left_join(get.imp(2)) %>% 
  left_join(get.imp(3)) %>% left_join(get.imp(4)) %>% 
  left_join(get.imp(5)) %>% left_join(get.imp(6)) %>% left_join(get.imp(7))
imp_by_models


#---------------------------------------------
# prediction on testing data set

# predicted prob.
get.predProb <- function(i, dat = test_imp) {
  tpd <- predict(model_list[[i]], newdata = dat, type = "prob")
  tpd <- tpd %>% select(yes)
  names(tpd) <- names(model_list[i])
  tpd
}
predictProb_test <- cbind(select(test_imp, aki:racewhite), get.predProb(1), get.predProb(2), 
                          get.predProb(3), get.predProb(4), get.predProb(5), 
                          get.predProb(6), get.predProb(7))
# predictProb_test %>% group_by(aki) %>% summarise_all(mean)


get.auc <- function(mod, dt = predictProb_test) {
  tp <- roc(dt$aki2, pull(dt, mod), ci = T)
  tp2 <-  tp$ci %>% as.numeric
  glue("{round(tp2[2], 3)} ({round(tp2[1], 3)}, {round(tp2[3], 3)})")
}
test_auc <- modelnames %>% sapply(get.auc) %>% data.frame()
test_auc

  # get pair-wise roc tests 
get.roc <- function(mod, dt = predictProb_test) {
  roc(dt$aki2, pull(dt, mod), ci = T)
}
test_roc <- modelnames %>% lapply(get.roc)
roc_p <- matrix(nrow = 7, ncol = 7)
for(i in 1:7) {
  for(j in 1:7) {
    roc_p[i, j] <- roc.test(test_roc[[i]], test_roc[[j]])$p.value
  }
}
roc_p <- roc_p %>% data.frame()
names(roc_p) <- modelnames


# predicted class
get.predClass <- function(i, dat = test_imp) {
  tpd <- predict(model_list[[i]], newdata = dat) %>% data.frame()
  names(tpd) <- names(model_list[i]) 
  tpd
}
predictClass_test <- cbind(select(test_imp, aki:racewhite), get.predClass(1), get.predClass(2), get.predClass(3), 
                           get.predClass(4), get.predClass(5), get.predClass(6), get.predClass(7))

get_cm <- function(mod, dat = predictClass_test) {
  tp <- confusionMatrix(pull(dat, mod), reference = dt$aki2)
  tp2 <- tp$byClass %>% t %>% data.frame()
  tp2 %>% mutate(model = mod, .before = Sensitivity)
}
test_cm <- modelnames %>% lapply(get_cm) %>% data.table::rbindlist()

test_metrics <- test_cm %>% mutate(AUROC = test_auc$., .after = model)
test_metrics

