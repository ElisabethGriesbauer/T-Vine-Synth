library(readr)
library(dplyr)
library(rvinecopulib)
library(data.table)
library(pROC)
library(ggplot2)
library(doBy)
library(tidyr)
library(latex2exp)
library(randomForest)
library(cowplot)
library(corrplot)

real_data <- read.csv("./data/preprocessed/real_data_I_d20.csv")
real_data <- as.data.table(real_data)

n <- dim(real_data)[1]
d <- dim(real_data)[2]

name <- "./figures/corr_real_data_I_d20.png"
png(name)
corrplot(cor(real_data[, -d, with = F]), method="color")
dev.off()

p_synth <- 1

attribute_names <- c(paste0(rep("X", d-1), c(1:(d-1))), "Y")
factor_position <- d
vine_estimation <- "par"

numerics <- attribute_names[-factor_position]
factors <- attribute_names[factor_position]

real_data[ , (factors) := lapply(.SD, factor, ordered = T), .SDcols = factors]
real_data[ , (numerics) := lapply(.SD, as.numeric), .SDcols = numerics]


primary <- c(numerics, factors)



# transforming the data to the unit cube
n_u_real <- real_data[, pseudo_obs(.SD), .SDcols = numerics]

c_u_real <- real_data[, lapply(X = .SD, FUN = rank, ties.method = "max"), .SDcols = factors]
c_u_real[, c(paste(factors, "_1", sep = "")) := lapply(X = .SD, FUN = function(x){
  (rank(x, ties.method = "min") - 1)
}), .SDcols = factors]

u_real <- cbind(n_u_real, c_u_real[, .SD/(n + 1)])

rm(n_u_real, c_u_real)

# remove col_names for fitting of vine
colnames(u_real) <- NULL

structure <- cvine_structure(order = 1:d, trunc_lvl = Inf)

vine_model <- vinecop(data = u_real, var_types = c(rep("c", length(numerics)), rep("d", length(factors))), 
                      par_method = "mle", family_set = "parametric", structure, nonpar_method = "linear", trunc_lvl = Inf)


for (t in c(1:(d-1))){
  
  vmodel <- truncate_model(vine_model, trunc_lvl = t)
  u_synth <- rvinecop(n * p_synth, vmodel) %>% data.table()
  names(u_synth) <- primary
  
  name <- paste0("./figures/simulated_real_data_estimated_correlation_on_synth_data_Cvine_trunc", t, ".png", sep = "")
  png(name)
  corrplot(cor(u_synth), method="color")
  dev.off()
  
}
