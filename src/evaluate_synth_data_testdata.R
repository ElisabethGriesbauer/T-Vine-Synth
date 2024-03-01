evaluate_synth_data_testdata <- function(real_data, synth_data, test_data, p_synth = 1, classifier = "log_reg"){
  
  # function that evaluates synthetic through classification and general measures
  # real_, synth_data: real and synthetic data of same size WITH RESPONSE AS LAST COLUMN
  # training_fraction: percentage of data to train classifier on
  # classifier: string to indicate which classifier to use for comparison; either: "log_reg", "rand_forest", "boosting", "neural_net"
  
  real_data <- real_data %>% as.data.frame()
  synth_data <- synth_data %>% as.data.frame()
  test_data <- test_data %>% as.data.frame()
  n <- dim(real_data)[1]
  d <- dim(real_data)[2]
  dx <- d - 1
  a <- 1
  
  
  #---------------------------------------------------------------------------
  ### defining classifiers (prediction should be numeric!)
  
  ## logistic regression WITHOUT regularization
  log.regression <- function(train_data, test_data, d){
    # function to perform binomial logistic regression task
    # train_data: data set with response variable called "Y"
    # test_data: data set with response variable called "Y"
    log.regression <- glm(Y ~ ., data = train_data, family = "binomial")
    predictions <- predict(log.regression, newdata = test_data, 
                           type = "response") %>% as.numeric() 
    
    return(predictions)
  }
  
  
  
  ## regularized logistic regression
  reg.log.regression <- function(train_data, test_data, a = 1, d){
    # function to perform binomial logistic regression task with regularization parameter a
    # a: elastic net mixing, i.e. regularization parameter; a = 1 is Lasso, a=0 is Ridge
    # train_data: data set with response variable called "Y"
    # test_data: data set with response variable called "Y"
    # d: number of covariates + 1 (response), i.e. no. of columns in the data sets
    # dx <- d-1
    cv.regularization <- cv.glmnet(x = as.matrix(train_data[, -d]), y = train_data$Y, 
                                   alpha = a, family = "binomial", standardize = T)
    
    reg.log.model <- glmnet(x = as.matrix(train_data[, -d]), y = train_data$Y, 
                            family = "binomial", alpha = a, standardize = T)  
    
    predictions <- predict(reg.log.model, newx = as.matrix(test_data[, -d]),
                           type="response", s=cv.regularization$lambda.1se)
    
    return(predictions)
  }
  
  
  
  ## random forest
  rand_forest <- function(train_data, test_data, d, max.ntree = 2000){
    # function to perform binomial random forest classification task
    # train_data: data set with response variable called "Y"
    # test_data: data set with response variable called "Y"
    # max.ntree: maximal number of ntree possible for the random forest (ntree is selected according to OBB)
    # d: number of covariates + 1 (response), i.e. no. of columns in the data sets
    train_data$Y <- train_data$Y %>% as.factor()
    # test_data$Y <- test_data$Y %>% as.factor()
    
    rf.err.rate <- function(x){
      fit <- randomForest(Y ~ ., data = train_data, ntree = x, mtry = floor(sqrt(d - 1)), replace = T, type = "classification")
      y <- mean(fit$err.rate[,1])
      return(y)
    }
    
    obb.err.mean <- vapply(X = seq(from=100, to=max.ntree, by=100), FUN = rf.err.rate, numeric(1))
    ntrees.opt <- seq(from=100, to=max.ntree, by=100)[which(min(unlist(obb.err.mean)) == unlist(obb.err.mean))]
    
    
    rf.model <- randomForest(Y ~ ., data = train_data, ntree = ntrees.opt, 
                             mtry = floor(sqrt(d - 1)), nodesize = 1, replace = T, type = "classification")
    predictions <- predict(rf.model, newdata = test_data[, -d], type = "response", 
                           norm.votes = T) %>% as.numeric()
    
    return(predictions)
  }
  
  
  ## boosting
  boosting <- function(train_data, test_data, d){
    # function to implement binary classification using gradient boosting
    # basic gradient boosting model with default parameter values for max.depth (6, depth of trees), eta (0.3, regularization?)
    # early stopping cannot be applied without validation set
    # no parameter tuning
    # train_data: data set with response variable called "Y"
    # test_data: data set with response variable called "Y"
    # d: no. of columns in the data sets
    
    dtrain <- xgb.DMatrix(data = as.matrix(train_data[, -d]), label = as.matrix(train_data[, d]))
    
    # 5-fold cross-validation for nrounds = no. of decision trees in boosting
    rounds = c(100, 300, 500, 700, 1000)
    cv_results = c(0)
    
    # Iterate over rounds and build one model for each value
    for (i in rounds){
      
      # perform cross-validation: cv_r
      cv_r = xgb.cv(data = dtrain, max.depth = 6, eta = 0.3, nthread = 1, objective = "binary:logistic", verbose = 0, nfold = 5, nrounds = i, metrics="auc")
      
      if( (cv_r$evaluation_log$test_auc_mean[i] - cv_results[length(cv_results)]) < 0.001 ){
        break
      }
      
      cv_results <- append(cv_results, cv_r$evaluation_log$test_auc_mean[i])
      
    }
    
    # -1 as I start with 0 in cv_results to be able to compare
    nrounds_cv <- rounds[which(cv_results == max(cv_results)) - 1]
    
    xgb.model <- xgb.train(data = dtrain, max.depth = 6, eta = 0.3, nthread = 1, nrounds = nrounds_cv, objective = "binary:logistic", verbose = 1)
    
    predictions <- predict(xgb.model, as.matrix(test_data[, -d]))
    
    return(predictions)
  }
  
  
  
  ## multi-layer perceptron (MLP) classifier:
  MLP <- function(train_data, test_data, d){
    # function to perform classification with a MLP
    # train_data: data set with response variable called "Y"
    # test_data: data set with response variable called "Y"
    
    
    if(all(train_data == test_data)){
      data_list <- list("inputsTrain" = as.matrix(train_data[, -d]), "targetsTrain" = as.matrix(as.numeric(train_data$Y) - 1), 
                        "inputsTest" = as.matrix(test_data[, -d]), "targetsTest" = as.matrix(as.numeric(test_data$Y) - 1))  
    } else {
      data_list <- list("inputsTrain" = as.matrix(train_data[, -d]), "targetsTrain" = as.matrix(train_data$Y), 
                        "inputsTest" = as.matrix(test_data[, -d]), "targetsTest" = as.matrix(as.numeric(test_data$Y) - 1))
    }
    
    norm_data <- normTrainingAndTestSet(x = data_list, dontNormTargets = TRUE, type = "norm") # "norm" is default
    
    MLP.model <- mlp(x = norm_data$inputsTrain, y = norm_data$targetsTrain, size = 5, maxit = 50) # not needed: , inputsTest = norm_data$inputsTest, targetsTest = norm_data$targetsTest
    
    predictions <- predict(MLP.model, newdata = norm_data$inputsTest)
    
    return(predictions)
  }
  #---------------------------------------------------------------------------
  ### defining measures
  
  ## variational information measure
  VI <- function(M){
    if (dim(M)[2] < 2) {
      VI <- "not defined"
    } else {
      
      real_0 <- sum(M[1,]); real_1 <- sum(M[2,])
      pred_0 <- sum(M[,1]); pred_1 <- sum(M[,2])
      n <- sum(M)
      
      if (any(c(real_0, real_1, pred_0, pred_1) == 0)){
        
        VI <- "not defined"
        
      } else {
        
        VI <- (-1) * real_0/n * log(real_0/n) - real_1/n * log(real_1/n) - pred_0/n * log(pred_0/n) - pred_1/n * log(pred_1/n) - 
          2 * ( M[1,1]/n * log(n * M[1,1]/(real_0 * pred_0)) + M[1,2]/n * log(n * M[1,2]/(real_0 * pred_1)) + M[2,1]/n * log(n * M[2,1]/(real_1 * pred_0)) + 
                  M[2,2]/n * log(n * M[2,2]/(real_1 * pred_1)) )
        
      } 
      
    }
    
    return(VI)
  }
  
  ## F-measure
  F_measure <- function(m){
    if (dim(m)[2] < 2){
      
      F <- 0
      
    } else if (sum(m[2,]) == 0 || sum(m[, 2]) == 0) {
      
      F <- Inf
      
    } else {
      
      p <- m[2,2]/(m[2,2]+m[1,2])
      r <- m[2,2]/(m[2,2]+m[2,1])
      F <- (2*p*r)/(p+r)  
      
    }
    
    return(F)
  }
  
  
  MCC <- function(M, predictions, tests){
    
    m_test <- tests %>% as.character() %>% as.numeric() %>% mean()
    m_r <- (predictions>=0.5) %>% as.numeric() %>% mean() 
    if (dim(M)[2] < 2) {
      mm <- 0
    } else {
      mm <- M[2,2]
    }
    MCC_ <- ( mm/n - m_test * m_r )/( sqrt( m_test * m_r * (1 - m_test) * (1 - m_r) ) )  
    
    return(MCC_)
  }
  
  #---------------------------------------------------------------------------
  ### real test data
  Y_r_test <- test_data$Y
  
  ### computing predictions on real data
  if (classifier == "log_reg"){
    Y_hat_r <- log.regression(train_data = real_data, test_data = test_data)
  } else if (classifier == "reg_log_reg"){
    Y_hat_r <- reg.log.regression(train_data = real_data, test_data = test_data, a = a, d = d)
  } else if (classifier == "rand_forest"){
    Y_hat_r <- rand_forest(train_data = real_data, test_data = test_data, d = d) - 1
  } else if (classifier == "boosting"){
    Y_hat_r <- boosting(train_data = real_data, test_data = test_data, d = d)
  } else {
    Y_hat_r <- MLP(train_data = real_data, test_data = test_data, d = d)
  }
  
  
  
  ### computing measures for real predictions wrt test data
  conf_matrix_real <- table("Y_real_test" = Y_r_test, "Y_hat_real" = as.numeric(Y_hat_r>=0.5))
  accuracy_real <- diag(conf_matrix_real) %>% sum() / sum(conf_matrix_real)
  F_real <- F_measure(conf_matrix_real)
  AUC_r <- auc(response = Y_r_test, predictor = as.numeric(Y_hat_r>=0.5))

  MCC_r <- MCC(M = conf_matrix_real, tests = Y_r_test, predictions = Y_hat_r)
  
  VI_r <- VI(conf_matrix_real)
  
  #---------------------------------------------------------------------------
  ### computing predictions and measures on synthetic data:  synthetic and directly comparing both
  
  results <- list()
  
  for (p in 1:p_synth) {
    
    synth_data_p <- synth_data[ ((p-1) * n + 1) : (p * n), ]
    
    if (classifier == "log_reg"){
      Y_hat_s <- log.regression(train_data = synth_data_p, test_data = test_data)
    } else if (classifier == "reg_log_reg"){
      Y_hat_s <- reg.log.regression(train_data = synth_data_p, test_data = test_data, a = a, d = d)
    } else if (classifier == "rand_forest"){
      Y_hat_s <- rand_forest(train_data = synth_data_p, test_data = test_data, d = d) - 1
    } else if (classifier == "boosting"){
      Y_hat_s <- boosting(train_data = synth_data_p, test_data = test_data, d = d)
    } else {
      Y_hat_s <- MLP(train_data = synth_data_p, test_data = test_data, d = d)
    }
    
    
    #---------------------------------------------------------------------------
    ### computing measures:  synthetic and directly comparing both
    
    ## confusion matrix
    conf_matrix_synth <- table("Y_real_test" = Y_r_test, "Y_hat_synth" = as.numeric(Y_hat_s>=0.5))
    conf_matrix_direct <- table("Y_hat_real" = as.numeric(Y_hat_r>=0.5), "Y_hat_synth" = as.numeric(Y_hat_s>=0.5))
    
    ## classification accuracy
    accuracy_synth <- diag(conf_matrix_synth) %>% sum() / sum(conf_matrix_synth)
    accuracy_direct <- diag(conf_matrix_direct) %>% sum() / sum(conf_matrix_direct)
    
    ## F-measure
    F_synth <- F_measure(conf_matrix_synth)
    F_direct <- F_measure(conf_matrix_direct)
    
    ## AUC
    # if ( all(as.numeric(Y_hat_s>=0.5) == 0) ||  all(as.numeric(Y_hat_r>=0.5) == 0)) {
    AUC_s <- auc(response = Y_r_test, predictor = as.numeric(Y_hat_s>=0.5))
    AUC_direct <- auc(response = as.numeric(Y_hat_r>=0.5), predictor = as.numeric(Y_hat_s>=0.5))
    # }
    
    
    ## MCC
    MCC_s <- MCC(M = conf_matrix_synth, tests = Y_r_test, predictions = Y_hat_s)
    MCC_direct <- MCC(M = conf_matrix_direct, tests = Y_hat_r, predictions = Y_hat_s)
    
    ## VI
    
    VI_s <- VI(conf_matrix_synth)
    VI_direct <- VI(conf_matrix_direct)
    
    results_p <- list("conf_matrix" = list("conf_matr_real" = conf_matrix_real, "conf_matr_synth" = conf_matrix_synth, "conf_matr_direct" = conf_matrix_direct),
                      "accuracy" = list("acc_real" = accuracy_real, "acc_synth" = accuracy_synth, "acc_direct" = accuracy_direct),
                      "F_measure" = list("F_real" = F_real, "F_synth" = F_synth, "F_direct" = F_direct),
                      "AUC" = list("AUC_real" = AUC_r, "AUC_synth" = AUC_s, "AUC_direct" = AUC_direct),
                      "MCC" = list("MCC_real" = MCC_r, "MCC_synth" = MCC_s, "MCC_direct" = MCC_direct),
                      "VI" = list("VI_real" = VI_r, "VI_synth" = VI_s, "VI_direct" = VI_direct))
    
    results[[p]] <- results_p
  }
  
  return(results)
}

