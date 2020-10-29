rm(list=ls())
setwd('C:/Users/Arathen/Desktop/Github Projects/Car Lemon/')

# Load Libraries
library(DataExplorer)
library(randomForest)
library(doParallel)
library(tidyverse)
library(caret)

# Parallel processing
cls = makeCluster(6)
registerDoParallel(cls)

# Reading in Data
train.pre <- read.csv("training.csv", stringsAsFactors = FALSE)
test.pre <- read.csv("test.csv", stringsAsFactors = FALSE)
# head(cbind(test.pre, "IsBadBuy" = NA))
to.clean <- rbind(cbind(train.pre[,-2], "IsBadBuy" = train.pre$IsBadBuy),
                  cbind(test.pre, "IsBadBuy" = NA))

# Gini index function
gini = function(
  truth=stop('supply actual probabilities'),
  preds=stop('supply predicted probabilities'),
  plot=FALSE
){ 
  n = length(c(truth))
  k = cbind(truth,preds,1:n)
  k = k[order(k[,2]),]
  #k = k[order(k[,3],decreasing=TRUE),] #Do we need this line? not quite sure what 'sortrows' does in the Matlab script. Is sorting by the third column for breaking ties? 
  
  if(plot){
    plot(cumsum(k[,1]/sum(k[,1])),type='l',col='blue')
    points(1:n/n,type='l')
  } 
  return(sum(1:n/n - cumsum(k[,1]/sum(k[,1])))/n)
  
}

# Function for caret
giniCaret <-  function(data, lev = NULL, model = NULL){
  n <-  nrow(data)
  k <- apply(data, 2, function(x) x %>% as.character %>% as.numeric)
  k <- k[order(k[,"pred"]),]
  c("gini" = sum(1:n/n - cumsum(k[,1]/sum(k[,1])))/n)
}

#converting respective columns into date/numeric/factor
#date
to.clean$PurchDate <- as.Date(to.clean$PurchDate, format = "%M/%d/%Y")

#numeric
to.numeric.names <- c("MMRAcquisitionAuctionAveragePrice", "MMRAcquisitionAuctionCleanPrice", "MMRAcquisitionRetailAveragePrice",
                      "MMRAcquisitonRetailCleanPrice", "MMRCurrentAuctionAveragePrice", "MMRCurrentAuctionCleanPrice", 
                      "MMRCurrentRetailAveragePrice", "MMRCurrentRetailCleanPrice")
to.clean[,to.numeric.names] <- sapply(to.clean[,to.numeric.names], as.numeric)
# to.clean[,to.numeric.names] <- lapply(to.clean[,to.numeric.names], function(x) {is.na(x) <- mean(x[!is.na(x)])})

#filling in missing values with average
for(i in to.numeric.names){
  to.clean[is.na(to.clean[,i]), i] <- mean(to.clean[,i], na.rm = TRUE)
}

#factor
#some clean up
to.clean$Transmission[to.clean$Transmission == ""] <- "NULL"
to.clean$Transmission[to.clean$Transmission == "Manual"] <- "MANUAL"

to.factor.names <- names(sapply(to.clean, is.character))[unname(sapply(to.clean,is.character))]
to.clean[,to.factor.names] <- lapply(to.clean[,to.factor.names], as.factor)
to.clean$BYRNO <- as.factor(to.clean$BYRNO)
to.clean$VNZIP1 <- as.factor(to.clean$VNZIP1)
to.clean$IsOnlineSale <- as.factor(to.clean$IsOnlineSale)
to.clean$IsBadBuy <- as.factor(to.clean$IsBadBuy)

str(to.clean)
plot_missing(to.clean)

to.model <- to.clean %>% select(RefId, IsBadBuy, VehicleAge, VehOdo, MMRAcquisitionAuctionAveragePrice,
                                VehBCost, WarrantyCost,
                                WheelTypeID, PRIMEUNIT, AUCGUART)
# Possibly remove size, include Prime

str(to.model)

# Preprocessing
## Dummy (Indicator) Variables
IVTrans <- dummyVars(IsBadBuy~.-RefId, data=to.model)
data.iv <- predict(IVTrans, newdata=to.model)  %>% as.data.frame() %>%
  bind_cols(., to.model %>% select(RefId, IsBadBuy))

## Principal Components Transformation
pcTrans <- preProcess(x=data.iv %>% select(-c(RefId, IsBadBuy)), method="pca")
data.pca <- predict(pcTrans, newdata=data.iv)
#plot_correlation(data.pca, type="continuous", cor_args=list(use="pairwise.complete.obs"))

## Center and Scaling
trans.cs <- preProcess(x=data.pca %>% select(-c(RefId, IsBadBuy)), method=c("center", "scale"))
data.cs <- predict(trans.cs, newdata=data.pca)
trans01 <- preProcess(x=data.cs %>% select(-c(RefId, IsBadBuy)), method="range",
                      rangeBounds=c(0,1))
data.01 <- predict(trans01, newdata=data.cs)

# Split into training and test sets
train <- data.01 %>% filter(!is.na(IsBadBuy))
test <- data.01 %>% filter(is.na(IsBadBuy))

## Try RF Model
rf <- randomForest(formula = IsBadBuy ~ .-RefId, data = train)
rf.preds <- data.frame(RefId=test$RefId, IsBadBuy=predict(rf, newdata=test, type="prob"))
rf.preds <- data.frame(RefId=rf.preds$RefId, IsBadBuy=rf.preds$IsBadBuy.1)
head(rf.preds, 25)
#write_csv(x=rf.preds, path="./UntunedRFPreds.csv")

## Fit a Boosted Gradient model
## Baseline model
grid_default <- expand.grid(
  nrounds = 100,
  max_depth = 6,
  eta = 0.3,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

train_control <- caret::trainControl(
  method = "none",
  verboseIter = FALSE, # no training log
  allowParallel = TRUE # FALSE for reproducible results 
)

xgb_base <- train(form=IsBadBuy~.-RefId,
                  data=(train),
                  method="xgbTree",
                  trControl=train_control,
                  tuneGrid=grid_default,
                  verbose=TRUE
)

## Next, start tuning hyperparameters
nrounds <- 500
tune_grid <- expand.grid(
  nrounds = seq(from = 200, to = nrounds, by = 50),
  eta = c(0.025, 0.05, 0.1),
  max_depth = c(2, 3, 4, 5),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

tune_control <- caret::trainControl(
  method = "cv", # cross-validation
  #summaryFunction=giniCaret,
  number = 3, # with n folds 
  #index = createFolds(tr_treated$Id_clean), # fix the folds
  verboseIter = FALSE, # no training log
  allowParallel = TRUE # FALSE for reproducible results 
)

xgb_tune <-train(form=IsBadBuy~.-RefId,
                 data=train,
                 method="xgbTree",
                 #metric="gini",
                 trControl=tune_control,
                 tuneGrid=tune_grid,
                 verbose=TRUE
)

# helper function for the plots
tuneplot <- function(x, probs = .90) {
  ggplot(x) +
    coord_cartesian(ylim = c(quantile(x$results$Accuracy, probs = probs), max(x$results$Accuracy))) +
    theme_bw()
}

tuneplot(xgb_tune)
xgb_tune$bestTune

## Next round of tuning
tune_grid2 <- expand.grid(nrounds = seq(from = 50, to = nrounds, by = 50),
                          eta = xgb_tune$bestTune$eta,
                          max_depth = ifelse(xgb_tune$bestTune$max_depth == 2,
                                             c(xgb_tune$bestTune$max_depth:4),
                                             xgb_tune$bestTune$max_depth - 1:xgb_tune$bestTune$max_depth + 1),
                          gamma = 0,
                          colsample_bytree = 1,
                          min_child_weight = c(1, 2, 3),
                          subsample = 1
)

xgb_tune2 <- caret::train(
  form=IsBadBuy~.-RefId,
  data=(train),
  method="xgbTree",
  trControl=tune_control,
  tuneGrid=tune_grid2,
  verbose=TRUE
)

tuneplot(xgb_tune2)
xgb_tune2$bestTune
max(xgb_tune$results$Accuracy)
max(xgb_tune2$results$Accuracy)

## Next tuning round
tune_grid3 <- expand.grid(
  nrounds = seq(from = 50, to = nrounds, by = 50),
  eta = xgb_tune$bestTune$eta,
  max_depth = xgb_tune2$bestTune$max_depth,
  gamma = 0,
  colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = c(0.5, 0.75, 1.0)
)

xgb_tune3 <- caret::train(
  form=IsBadBuy~.-RefId,
  data=(train),
  method="xgbTree",
  trControl=tune_control,
  tuneGrid=tune_grid3,
  verbose=TRUE
)

tuneplot(xgb_tune3, probs = .95)
xgb_tune3$bestTune
max(xgb_tune$results$Accuracy)
max(xgb_tune2$results$Accuracy)
max(xgb_tune3$results$Accuracy)

## Tuning the Gamma
tune_grid4 <- expand.grid(
  nrounds = seq(from = 50, to = nrounds, by = 50),
  eta = xgb_tune$bestTune$eta,
  max_depth = xgb_tune2$bestTune$max_depth,
  gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0),
  colsample_bytree = xgb_tune3$bestTune$colsample_bytree,
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = xgb_tune3$bestTune$subsample
)

xgb_tune4 <- caret::train(
  form=IsBadBuy~.-RefId,
  data=(train),
  method="xgbTree",
  trControl=tune_control,
  tuneGrid=tune_grid4,
  verbose=TRUE
)

tuneplot(xgb_tune4)
xgb_tune4$bestTune
max(xgb_tune$results$Accuracy)
max(xgb_tune2$results$Accuracy)
max(xgb_tune3$results$Accuracy)
max(xgb_tune4$results$Accuracy)

## Reduce learning rate
tune_grid5 <- expand.grid(
  nrounds = seq(from = 100, to = 10000, by = 100),
  eta = c(0.01, 0.015, 0.025, 0.05, 0.1),
  max_depth = xgb_tune2$bestTune$max_depth,
  gamma = xgb_tune4$bestTune$gamma,
  colsample_bytree = xgb_tune3$bestTune$colsample_bytree,
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = xgb_tune3$bestTune$subsample
)

xgb_tune5 <- caret::train(
  form=IsBadBuy~.-RefId,
  data=(train),
  method="xgbTree",
  trControl=tune_control,
  tuneGrid=tune_grid5,
  verbose=TRUE
)

tuneplot(xgb_tune5)
xgb_tune5$bestTune
max(xgb_tune$results$Accuracy)
max(xgb_tune2$results$Accuracy)
max(xgb_tune3$results$Accuracy)
max(xgb_tune4$results$Accuracy)
max(xgb_tune5$results$Accuracy)

## Fit the model and predict
final_grid <- expand.grid(
  nrounds = xgb_tune5$bestTune$nrounds,
  eta = xgb_tune5$bestTune$eta,
  max_depth = xgb_tune5$bestTune$max_depth,
  gamma = xgb_tune5$bestTune$gamma,
  colsample_bytree = xgb_tune5$bestTune$colsample_bytree,
  min_child_weight = xgb_tune5$bestTune$min_child_weight,
  subsample = xgb_tune5$bestTune$subsample
)

xgb_model <- caret::train(
  form=IsBadBuy~.-RefId,
  data=(train),
  method="xgbTree",
  trControl=tune_control,
  tuneGrid=final_grid,
  verbose=TRUE
)

xgb.preds <- data.frame(RefId=test$RefId, IsBadBuy=predict(xgb_model, newdata=test, type="prob"))
xgb.preds <- data.frame(RefId=xgb.preds$RefId, IsBadBuy=xgb.preds$IsBadBuy.1)
head(xgb.preds, 25)
#write_csv(x=xgb.preds, path="./XGBPredictions.csv")


# GBM
gbm_model <- train(form=IsBadBuy~.-RefId,
                   data=(train),
                   method="gbm",
                   trControl=tune_control,
                   tuneGrid=expand.grid(n.trees=c(150, 200, 250),
                             interaction.depth=c(2,3,4),
                             shrinkage=c(.01,.05,.1),
                             n.minobsinnode=c(7,9,10,11,13)),
                   #metric="gini",
                   verbose=TRUE
)

gbm.preds <- data.frame(RefId=test$RefId, IsBadBuy=predict(gbm_model, newdata=test, type="prob"))
gbm.preds <- data.frame(RefId=gbm.preds$RefId, IsBadBuy=gbm.preds$IsBadBuy.1)
head(gbm.preds, 25)
#write_csv(x=gbm.preds, path="./GBMPredictions.csv")