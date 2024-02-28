library(MASS)
library(dplyr)
library(pROC)
library(Matrix)


#############################
#### SIMULATED REAL DATA ####
#############################

# We simulate real data with d = 20 plus Y, n = 1000. The underlying distribution 
# is a multivariate normal (MN) per class, i.e. on MN(Y=0) and MV(Y=1). For both 
# MNs we have a block dependence structure. (X_1, ..., X_5) and (X_6, ..., X_10) have dependence among 
# each other and (X_11, ..., X_20, Y) have dependence among each other. Between
# MN(Y=0) and MN(Y=1) the distribution of the first two blocks remains (almost)
# the same, in the block (X_11, ..., X_20, Y) the marginals as well as Sigma changes.
# Note that increasing d(mu_1, mu_2) corresponds to main effects, increasing 
# d(Sigma_1, Sigma_2) corresponds to quadratic effects.


d = 20

# dimensions of blocks
d1 = 5
d2 = 5
d3 = 10

# function to randomly generate correlation matrices
sample_corr_matrix <- function(d){
  lower_triangular <- matrix(0, d, d)
  for (i in 1:d) {
    for (j in 1:(i - 1)) {
      lower_triangular[i, j] <- runif(1, -0.3, 0.3)  # Generate random values between -1 and 1
    }
  }
  
  # Set the diagonal elements to 1
  diag(lower_triangular) <- 1
  # Convert the lower triangular matrix to a symmetric matrix
  cor_matrix <- lower_triangular %*% t(lower_triangular)
  
  for (i in 1:d){
    for (j in 1:d) {
      cor_matrix[i,j] <- cor_matrix[i,j]/sqrt(cor_matrix[i,i] * cor_matrix[j,j])
    }
  }
  
  
  return(cor_matrix) 
}


## defining distributions ##
############################
#-------------------------------------------------------------------------------
### Y=0
set.seed(1234)
sigmas_0_I  <- rgamma(d, shape = 5, scale = 1)
mu_0_I      <- runif(d, max = 30, min = -5)

cor1 <- sample_corr_matrix(d1)
cor2 <- sample_corr_matrix(d2)
cor3 <- sample_corr_matrix(d3)

Corr_0_I <- bdiag(cor1, cor2, cor3)

Corr_0_I <- Corr_0_I %*% t(Corr_0_I)
Sigma_0_I   <- diag(sqrt(sigmas_0_I)) %*% Corr_0_I %*% diag(sqrt(sigmas_0_I))



### Y=1
new_sigmas <- rgamma(d3, shape = 3.5, scale = 1)

delta_new_mus <- runif(d3, max = 1.5, min = -1.5)
delta_new_sigmas <- runif(d3, max = 1, min = -1)

mu_1_I <- mu_0_I
sigmas_1_I  <- sigmas_0_I


mu_1_I[c((d1+d2+1):d)] <- mu_1_I[c((d1+d2+1):d)] + delta_new_mus
sigmas_1_I[c((d1+d2+1):d)] <- sigmas_1_I[c((d1+d2+1):d)] + delta_new_sigmas


new_cor3 <- sample_corr_matrix(d3)

Corr_1_I <- bdiag(cor1, cor2, new_cor3)
Corr_1_I <- Corr_1_I %*% t(Corr_1_I)
Sigma_1_I   <- diag(sqrt(sigmas_1_I)) %*% Corr_1_I %*% diag(sqrt(sigmas_1_I))



## sampling ##
##############
#-------------------------------------------------------------------------------
set.seed(123)
real_data_I_0 <- mvrnorm(n = 500, mu = mu_0_I, Sigma = Sigma_0_I) %>% cbind(rep(0, 500))
real_data_I_1 <- mvrnorm(n = 500, mu = mu_1_I, Sigma = Sigma_1_I) %>% cbind(rep(1, 500))

real_data_I <- rbind(real_data_I_0, real_data_I_1)
rm(real_data_I_0, real_data_I_1)


# naming
names <- c(paste0(rep("X", d), c(1:d), sep = ""), "Y")
colnames(real_data_I) <- names
#-------------------------------------------------------------------------------





## testing the data ##
######################
#-------------------------------------------------------------------------------
set.seed(456)
train_split <-  sort(sample(1000, floor(1000 * 0.7)))
data <- as.data.frame(real_data_I)
data$Y <- data$Y %>% as.factor()

log.regression <- glm(Y ~ ., data = data, family = "binomial", subset = train_split)
predictions <- predict(log.regression, newdata = data[-train_split, ], 
                       type = "response") %>% as.numeric() 


table("Y" = data$Y[-train_split], "pred_Y" = (predictions >= 0.5) %>% as.numeric()) / 300

AUC <- auc(response = data$Y[-train_split], predictor = as.numeric(predictions))
print("AUC:"); AUC




## writing the data ##
######################
#-------------------------------------------------------------------------------
write_s_r_data <- function(s_data, dir, name){
  s_data$Y <- s_data$Y %>% as.numeric() %>% as.integer
  write.csv(s_data, paste(name, ".csv", sep = ""), row.names = F)
}

write_s_r_data(real_data_I %>% as.data.frame(), name = "real_data_I_d20")


## read the data ##
###################
#-------------------------------------------------------------------------------

library(readr)
real_data_I <- read.csv("real_data_I_d20.csv")


## sample target patients ##
############################
#-------------------------------------------------------------------------------
set.seed(123456)
targets <- sample(1000, 4); targets




## plotting corr ##
############################
#-------------------------------------------------------------------------------
library(corrplot)

corrplot(cor(real_data_I), method="color")

real_data_I %>% summary()



## generating test data ##
##########################
#-------------------------------------------------------------------------------
set.seed(789)
real_data_I_0_test <- mvrnorm(n = 125, mu = mu_0_I, Sigma = Sigma_0_I) %>% cbind(rep(0, 125))
real_data_I_1_test <- mvrnorm(n = 125, mu = mu_1_I, Sigma = Sigma_1_I) %>% cbind(rep(1, 125))

real_data_I_test <- rbind(real_data_I_0_test, real_data_I_1_test)
rm(real_data_I_0_test, real_data_I_1_test)

# naming
names <- c(paste0(rep("X", d), c(1:d), sep = ""), "Y")
colnames(real_data_I_test) <- names

corrplot(cor(real_data_I_test), method="color")

# writing
write_s_r_data(real_data_I_test %>% as.data.frame(), name = "real_data_I_d20_test")
