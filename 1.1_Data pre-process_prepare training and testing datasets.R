library(tidyverse)
library(tidymodels)
library(caret)
library(skimr)
library(glue)

# read input data
dat0 <- read_csv("0_input data/sph6004_assignment1_data.csv")
dat0$aki %>% table %>% prop.table()*100
dat0 %>% skim
dat0 <- dat0 %>% mutate(id = as.character(id))

# naive cleaning of outlines likely due to data entry error or meant to be missing (eg, 999999 for glucose)
# winserise outlines for clinical measures (given no clinical knowledge for now..)
fun.winserise <- function(x) {
  # upper <- mean(x, na.rm = T) + 5*sd(x, na.rm = T)
  # lower <- mean(x, na.rm = T) - 5*sd(x, na.rm = T)
  upper <- quantile(x, 0.99, na.rm = T)
  lower <- quantile(x, 0.01, na.rm = T)
  x <- ifelse(x > upper, upper, x)
  x <- ifelse(x < lower, lower, x)
  x
}
dat1 <- dat0 %>% mutate_at(vars(heart_rate_min:weight_admit), fun.winserise)

# remove variables that contains >20% missing data
tp_var <- dat1 %>% skim %>% data.frame() %>% filter(complete_rate >=.8) %>% pull(skim_variable) # retain 65 variables
dat2 <- dat1 %>% select(all_of(tp_var))
dat2 %>% skim

# clean variable race
race0 <- dat0$race %>% tolower
race0 %>% table
dat2 <- dat2 %>% mutate(race = tolower(race)) %>% 
  mutate(race = case_when(.default = race,
                          grepl("black", race) ~ "black",
                          grepl("white|portuguese|hispanic|latino|south american", race) ~ "0_white",
                          grepl("asian", race) ~ "asian",
                          grepl("unable to obtain|patient declined to answer", race) ~ "unknown",
                          grepl("multiple race/ethnicity|american indian|native hawaiian", race) ~ "other")) %>% 
  mutate(race = case_when(.default = race,
                          grepl("unknown|other", race) ~ "unknown/other"))
dat2$race %>% table %>% prop.table()*100
  # further reduce to 2 groups for now: white and others 
dat2 <- dat2 %>% mutate(race = ifelse(race %in% c("asian", "black", "unknown/other"), "others", "white"))
table(race0, dat2$race)

# create dummy variables for categorical variables race and gender
dat3 <- dat2 %>% select(-id)
dummy <- dummyVars(~., data = dat3, fullRank = T)
dat3 <- predict(dummy, newdata = dat3) %>% data.frame()

# check zero-variance predictors
nearZeroVar(dat3) # no issue

# check correlated predictors 
# use threshold 0.8 to remove highly correlated variables (to avoid multicollinearity)
tpcor <- dat3 %>% select(-aki) %>% cor(use = "pairwise.complete.obs")
summary(tpcor[upper.tri(tpcor)])
sum(abs(tpcor[upper.tri(tpcor)]) > .8)
highcor <- findCorrelation(tpcor, cutoff = .8)
dat4 <- dat3 %>% select(-aki)
dat4 <- dat4[, -highcor]
dat4 <- dat4 %>% mutate(aki = dat3$aki, .before = genderM)

# split training and test data: splitting based on the outcome
set.seed(202403)
train_idx <- createDataPartition(dat4$aki, p = 0.7, list = F, times = 1)
train0 <- dat4[train_idx, ]
test0 <- dat4[-train_idx, ]

# scale, center for numeric predictors variables
# impute for missing data using K-nearest neighbors based on the predictors
# use training data to obtain the scaling & imputation parameters
scale_impute <- preProcess(select(train0, -aki), method = c("center", "scale", "knnImpute"))
train_imp <- predict(scale_impute, select(train0, -aki))
test_imp <- predict(scale_impute, select(test0, -aki))
  # add back the outcome variable aki
train_imp <- train_imp %>% mutate(aki = train0$aki, aki2 = ifelse(train0$aki>0, 'yes', 'no') %>% factor(), .before = genderM)
test_imp <- test_imp %>% mutate(aki = test0$aki, aki2 = ifelse(test0$aki>0, 'yes', 'no') %>% factor() , .before = genderM)

save(train_imp, test_imp, file = "0_rda/1.1_pre-filter_center_scaled_imputed training and testing dataset_48predictors.rda")




# check if can future reduce the nr predictors using traditional (no repeated CV) univariate logistic regression or stepwise logistic regresson
vars <- train_imp %>% select(-aki, -aki2) %>% names
p_glm <- vars %>% lapply(function(x) {
  fit <- eval(parse(text = glue("glm(aki2 ~ {x}, data = train_imp, family = 'binomial')")))
  tidy(fit) %>% filter(term != "(Intercept)")
}) %>% data.table::rbindlist() %>% select(-std.error, -statistic)

p_glm %>% skim
sig_univariate <- p_glm %>% filter(p.value <0.05) %>% pull(term)

dat <- train_imp %>% select(-aki)
fit <- eval(parse(text = glue("glm(aki2 ~ ., data = dat, family = 'binomial')")))
fitstep <- stats::step(fit)
sig_stepwise <- summary(fitstep)$coefficients %>% rownames()

setdiff(sig_univariate, sig_stepwise[-1])
setdiff(sig_stepwise[-1], sig_univariate)
union(sig_stepwise[-1], sig_univariate) %>% length # all 48 predictors would be retained either based on stepwise regression or univariate regression.


# # get participant characteristics
# datx <- dat2 %>% mutate(subset = ifelse(row_number() %in% train_idx, "training", "testing"), .before = id) %>% 
#   select(subset:admission_age) %>% mutate(aki2 = ifelse(aki>0, 1, 0), aki = aki %>% as.character())
# crosstable::crosstable(datx, cols = c("gender", "race", "aki", "aki2"), by = "subset", percent_digits = 0, 
#            percent_pattern = "{n} ({p_col})") 
# datx %>% group_by(subset) %>% skim

