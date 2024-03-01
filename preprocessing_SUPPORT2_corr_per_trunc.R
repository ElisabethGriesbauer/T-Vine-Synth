library(readr)
library(dplyr)
library(corrplot)
library(glmnet)
library(pROC)
library(ggplot2)
library(data.table)
library(randomForest)
library(stats)

library(readr)
# https://archive.ics.uci.edu/dataset/880/support2
raw_data <- read_csv("./data/raw/raw_support2.csv")
raw_data <- data.table(raw_data)

factor_positions <- c(3,4,5,8,9,11,12,18,26,28,29,45,47)

# transforming to factors
# raw_data[, (factor_positions) := lapply(.SD, as.factor), .SDcols = factor_positions]

summary(raw_data)



## sub-setting the columns of support2 data
exclude <- c("id", "sex", "dzgroup", "dzclass", "edu", "income", "avtisst", "race", 
             "prg6m", "dnr", "glucose", "glucose", "urine", "adlp", "sfdm2")

sub_data <- raw_data[, -which(names(raw_data) %in% exclude), with = FALSE]

# exclude missing values
sub_complete <- sub_data[complete.cases(sub_data), ]

# reorder
responses <- c("death", "hospdead", "ca")
sub_complete <- sub_complete[, c(setdiff(names(sub_complete), responses), responses), with = FALSE]
cleaned_real_data <- sub_complete[, -c(3, 33, 34)]

# also remove dementia and diabetes
cleaned_real_data[, dementia := NULL]
cleaned_real_data[, diabetes := NULL]
cleaned_real_data[, adls := NULL]
cleaned_real_data[, adlsc := NULL]


# create test data, i.e. 20% of the data that is not touched before utility evaluation
set.seed(3527)
test_samples <- sample(1:(dim(cleaned_real_data)[1]), floor(0.2*dim(cleaned_real_data)[1]), replace = F)

test_data <- cleaned_real_data[test_samples, ]
real_data <- cleaned_real_data[-test_samples, ]

names(real_data)[length(names(real_data))] <- "Y"
names(test_data)[length(names(test_data))] <- "Y"




# writing data
write_s_r_data <- function(s_data, dir, name){
  s_data$Y <- s_data$Y %>% as.numeric() %>% as.integer()
  write.csv(s_data, paste(name, ".csv", sep = ""), row.names = F)
}

write_s_r_data(real_data %>% as.data.frame(), name = "./data/preprocessed/real_support2_small")
write_s_r_data(test_data %>% as.data.frame(), name = "./data/preprocessed/test_support2_small")


n <- dim(real_data)[1]
d <- dim(real_data)[2]


# plotting correlation matrix of synthetic data generated with a Cvine for different truncaiotn levels
p_synth <- 1
trunc_levels = c(1, seq(5,20,5), 26)



attribute_names <- colnames(real_data)
factor_position <- c(d)
vine_estimation <- "par"

real_data$Y <- real_data$Y %>% as.factor()
real_data <- as.data.table(real_data)

numerics <- attribute_names[-factor_position]
factors <- attribute_names[factor_position]

primary <- c(numerics, factors)

# create ORDERED factors:
real_data[ , (factors) := lapply(.SD, factor, ordered = T), .SDcols = factors]
real_data[ , (numerics) := lapply(.SD, as.numeric), .SDcols = numerics]

interim_data <- copy(real_data)
u_real <- copy(real_data)

u_real[, (numerics) := pseudo_obs(.SD), .SDcols = numerics]
u_real[, (factors) := lapply(X = .SD, FUN = rank, ties.method = "max"), .SDcols = factors]

custom_rank <- function(x){ rank(x, ties.method = "min") - 1}

for (col_name in factors) {
  new_col_name <- paste0(col_name, "_1")
  u_real[, (new_col_name) := custom_rank(get(col_name))]
}

not_numerics <- setdiff(names(u_real), numerics)

u_real[, (not_numerics) := lapply(.SD, function(x) x/(n + 1)), .SDcols = not_numerics]

colnames(u_real) <- NULL

structure <- cvine_structure(order = 1:d)
vine_model <- vinecop(data = u_real, var_types = c(rep("c", length(numerics)), rep("d", length(factors))), 
                      par_method = "mle", family_set = "parametric", structure, nonpar_method = "linear")


for (t in 1:length(trunc_levels)) {
  
  v_model <- truncate_model(vine_model, trunc_lvl = trunc_levels[t])
  # simulating synthetic data on the unit cube from the vine
  u_synth <- rvinecop(n * p_synth, v_model) %>% data.table()
  names(u_synth) <- primary
  
  synth_data <- data.table(matrix(NA, nrow = p_synth*n, ncol = length(colnames(real_data)), dimnames = list(NULL, colnames(real_data))))
  
  for (x in numerics){
    synth_data[, (x) := quantile(real_data[[x]], probs = u_synth[[x]], type = 8)]
  }
  
  for (x in factors){
    synth_data[, (x) := quantile(real_data[[x]], probs = u_synth[[x]], type = 3)]
  }
  
  name <- paste0("estKendallstau_synthdata_Cvine_trunc", trunc_levels[t], "_support2Small.png", sep = "")
  png(name)
  corrplot(cor(synth_data[, ..numerics], method = "kendall"), method="color")
  dev.off()
  
}


name <- paste0("./figures/estKendallstau_realsupport2Small.png", sep = "")
png(name)
corrplot(cor(real_data[, ..numerics], method = "kendall"), method="color")
dev.off()

