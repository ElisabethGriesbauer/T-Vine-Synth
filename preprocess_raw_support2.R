library(readr)
library(dplyr)
library(corrplot)
library(glmnet)
library(pROC)
library(ggplot2)
library(data.table)
library(randomForest)
library(stats)

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