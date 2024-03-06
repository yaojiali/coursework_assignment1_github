library(tidyverse)
library(caret)
library(skimr)
library(doParallel) 
load("0_rda/1.1_pre-filter_center_scaled_imputed training and testing dataset_48predictors.rda")


# feature selection for ML methods without built-in feature selection.
set.seed(202403)
rfFuncs$summary <- twoClassSummary
ctrl <- rfeControl(functions = rfFuncs, # functions for model fitting, prediction and variable importance 
                   method = "repeatedcv", 
                   repeats = 2, 
                   number = 5, 
                   verbose = FALSE)

subsets <- c(seq(1, 48, 3)) # preliminary check that 1:10 has low ROCAUC..
cl <- makeCluster(detectCores())
registerDoParallel(cl)
fit_ref <- rfe(x = select(train_imp, -aki, -aki2),   # time consuming to run...
               y = train_imp$aki2,
               sizes = subsets,
               rfeControl = ctrl, 
               metric = "ROC")
# save(fit_ref, file = "0_rda/1.2 recursive feasture elimination model.rda")

fit_ref # the highest ROCAUC is at when all 48 features were included.
plot(fit_ref)
