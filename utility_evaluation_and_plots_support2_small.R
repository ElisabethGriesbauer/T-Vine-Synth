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


source("./src/evaluate_synth_data_testdata.R")
source("./src/generate_synth_data.R")
source("./src/making_tables.R")
real_data <- read.csv("./data/preprocessed/real_support2_small.csv")
test_data <- read.csv("./data/preprocessed/test_support2_small.csv")



n <- dim(real_data)[1]
d <- dim(real_data)[2]

p_synth <- 50
trunc_levels <- c(1, seq(5,20,5), 26)

attribute_names <- colnames(real_data)
factor_position <- c(d)
vine_estimation <- "par"

real_data$Y <- real_data$Y %>% as.factor()
real_data <- as.data.table(real_data)

test_data$Y <- test_data$Y %>% as.factor()
test_data <- as.data.table(test_data)

set.seed(456)

list_synth_data <- generate_synth_data_all_truncs(real_data, p_synth, attribute_names, factor_position, vine_estimation, vine_topology = "Cvine", trunc_levels = trunc_levels)

rm(factor_position, vine_estimation, attribute_names)
saveRDS(list_synth_data, file = "list_synthdata_realsupport2_small_Cvine_1_5_10_15_20_25_28.RData")

#utility
set.seed(4562)
utility_per_trunc <- list()
for (t in 1:length(list_synth_data)){
  utility_per_trunc[[t]] <- evaluate_synth_data_testdata(real_data, list_synth_data[[t]], test_data, p_synth, classifier = "rand_forest")
  names(utility_per_trunc)[t] <- names(list_synth_data)[t]
}

names(utility_per_trunc) <- names(list_synth_data)

# saving
saveRDS(utility_per_trunc, file = "utility_realsupport2_small_Cvine_randForest_p50_testdata.RData")
saveRDS(list_synth_data, file = "list_synthdata_realsupport2_small_Cvine_1_5_10_15_20_25_28.RData")




#-------------------------------------------------------------------------------

modelname <- "Cvine"

#rearranging
names(utility_per_trunc) <- paste0(modelname, "_trunc_", names(utility_per_trunc))
resultsCvine <- making_tables(utility_per_trunc, as_matrix = T)
models <- colnames(resultsCvine)
cnames <- row.names(resultsCvine)
resultsCvine <- t(resultsCvine)
colnames(resultsCvine) <- cnames
row.names(resultsCvine) <- NULL
resultsCvine <- resultsCvine %>% as.data.frame()
resultsCvine$model <- models

rm(models, cnames)



# real test data utility as a reference
set.seed(456)
data <- as.data.frame(real_data)
data$Y <- data$Y %>% as.factor()

t_data <- as.data.frame(test_data)
t_data$Y <- t_data$Y %>% as.factor()

rf.err.rate <- function(x){
  fit <- randomForest(Y ~ ., data = data, ntree = x, mtry = floor(sqrt(d - 1)), replace = T, type = "classification")
  y <- mean(fit$err.rate[,1])
  return(y)
}

obb.err.mean <- vapply(X = seq(from=100, to=2000, by=100), FUN = rf.err.rate, numeric(1))
ntrees.opt <- seq(from=100, to=2000, by=100)[which(min(unlist(obb.err.mean)) == unlist(obb.err.mean))]


rf.model <- randomForest(Y ~ ., data = data, ntree = ntrees.opt, 
                         mtry = floor(sqrt(d - 1)), nodesize = 1, replace = T, type = "classification")
predictions <- predict(rf.model, newdata = t_data[, -d], type = "response", 
                       norm.votes = T) %>% as.numeric()

AUCreal <- auc(response = t_data$Y, predictor = as.numeric(predictions)- 1)
print("AUC:"); AUCreal



#plotting

resultsCvine <- resultsCvine %>% separate(model, into = c("model", "t", "trunc"), sep = "_")
resultsCvine$t <- NULL
resultsCvine[, c("model", "trunc")] <- lapply(resultsCvine[, c("model", "trunc")], factor)



subdf <- resultsCvine[resultsCvine$model == "Cvine",]
med <- summaryBy(AUC_synth ~ (trunc), data = subdf, FUN = list(median)) %>% as.data.frame()

gCvine <- ggplot(subdf, aes(x = trunc, y = AUC_synth)) +
  geom_boxplot(width = 0.4, linewidth = 1.5, position=position_dodge(0.6), fill = '#0099FF') +
  labs(x ="truncation level", y = TeX(r'($AUC(y^*, \hat{w}^*)$)')) +
  theme_minimal() +
  theme(text = element_text(size = 30), legend.position = "bottom",
        legend.box = "vertical", plot.caption = element_text(hjust = 0),
        axis.text.x = element_text(size=30),
        axis.text.y = element_text(size=30),
        legend.text = element_text(size = 40),
        plot.margin=unit(c(2,0,0.5,0.5), 'cm')) +
  scale_y_continuous(breaks = seq(from = 0.5, to = 0.8, by = 0.1), limits = c(0.45, 0.8)) +
  geom_path(data = med, aes(x = trunc, y = AUC_synth.median), color = '#0099FF', group=1, size = 2) +
  scale_x_discrete(labels = c(paste(unique(subdf$trunc)[-length(unique(subdf$trunc))]), 'no')) +
  geom_hline(yintercept=AUCreal, alpha = 0.8, color = '#FF6633', size=2) +
  geom_text(x = 1.5, y = (AUCreal+0.07), label = TeX(r'($AUC(y^*, \hat{y}^*)$)'), size = 10, color = '#FF6633') +
  panel_border()






### -----------------------------------------------------------
### utility of PrivBayes

synth_data_competitors <- list()

for (i in 1:3){
  synth_data <- read.csv(paste0(dirname, "synth_data_PrivBayes", c("01", "1", "5")[i], "_real_support2_small.csv"))
  synth_data$Y <- synth_data$Y %>% as.factor()
  synth_data_competitors[[i]] <- synth_data
}

names(synth_data_competitors) <- paste("PrivBayes_", c("01", "1", "5"), sep = "")

set.seed(472)
util_competitors <- list()
for (i in 1:length(synth_data_competitors)){
  util_competitors[[i]] <- evaluate_synth_data_testdata(real_data, synth_data_competitors[[i]], test_data, p_synth, classifier = "rand_forest")
}

names(util_competitors) <- names(synth_data_competitors)

# saving (and reading)
saveRDS(util_competitors, file = "utility_realsupport2_small_PrivaBayes_eps_01_1_5_randForest_p50_testdata.RData")



resultsPrivBayes <- making_tables(util_competitors, as_matrix = T)
models <- colnames(resultsPrivBayes)
cnames <- row.names(resultsPrivBayes)
resultsPrivBayes <- t(resultsPrivBayes)
colnames(resultsPrivBayes) <- cnames
row.names(resultsPrivBayes) <- NULL
resultsPrivBayes <- resultsPrivBayes %>% as.data.frame()
resultsPrivBayes$model <- models

rm(models, cnames)


resultsPrivBayes <- resultsPrivBayes %>% separate(model, into = c("model", "eps"), sep = "_")
resultsPrivBayes$eps[resultsPrivBayes$eps == "01"] <- "0.1"
resultsPrivBayes[, c("model", "eps")] <- lapply(resultsPrivBayes[, c("model", "eps")], factor)

df <- subset(resultsPrivBayes, model == "PrivBayes")  
medPB <- summaryBy(AUC_synth ~ eps, data = df, FUN = list(median)) %>% as.data.frame()

gPB <- ggplot(df, aes(x = eps, y = AUC_synth)) + 
  geom_boxplot(width = 0.4, linewidth = 1.5, position=position_dodge(0.6), fill = '#0099FF') +
  # scale_fill_manual(labels = c("PrivBayes"), values = c( '#FF6633')) +
  labs(x =TeX(r'($\epsilon$)'), y = "") + # TeX(r'($AUC(y^*, \hat{w}^*)$)')
  theme_minimal() +
  theme(text = element_text(size = 40), legend.position = "bottom",
        legend.box = "vertical", plot.caption = element_text(hjust = 0),
        axis.text.x = element_text(size=30),
        axis.text.y = element_blank(),
        legend.text = element_text(size = 40),
        plot.margin=unit(c(2,0.5,0.5,-1), 'cm')) +
  geom_path(data = medPB, aes(x = eps, y = AUC_synth.median), color = '#0099FF', group=1, size = 2) +
  scale_y_continuous(breaks = seq(from = 0.5, to = 0.8, by = 0.1), limits = c(0.45, 0.8)) +
  geom_hline(yintercept=AUCreal, alpha = 0.8, color = "#FF6633", size=2) +
  panel_border()


### -----------------------------------------------------------
### utility of CTGAN and TVAE
synth_data_ctgan <- read.csv("synth_data_CTGAN_real_support2_small.csv")
synth_data_tvae <- read.csv("synth_data_TVAE_real_support2_small.csv")

synth_data_ctgan$Y <- synth_data_ctgan$Y %>% as.factor()
synth_data_tvae$Y <- synth_data_tvae$Y %>% as.factor()

synth_data_competitors <- list("CTGAN" = synth_data_ctgan, "TVAE" = synth_data_tvae)

set.seed(123)
util_competitors <- list()
for (i in 1:length(synth_data_competitors)){
  util_competitors[[i]] <- evaluate_synth_data_testdata(real_data, synth_data_competitors[[i]], test_data, p_synth, classifier = "rand_forest")
}

names(util_competitors) <- names(synth_data_competitors)

# saving
saveRDS(util_competitors, file = "utility_realsupport2_small_CTGAN_TVAE_randForest_p50_testdata.RData")
utilityCT <- util_competitors



#rearranging
resultsCT <- making_tables(utilityCT, as_matrix = T)
models <- colnames(resultsCT)
cnames <- row.names(resultsCT)
resultsCT <- t(resultsCT)
colnames(resultsCT) <- cnames
row.names(resultsCT) <- NULL
resultsCT <- resultsCT %>% as.data.frame()
resultsCT$model <- models

rm(models, cnames)



gCT <- ggplot(resultsCT, aes(x = model, y = AUC_synth)) + 
  geom_boxplot(width = 0.4, linewidth = 1.5, position=position_dodge(0.6), fill = '#0099FF') +
  labs(x ='', y = '') +
  theme_minimal() +
  theme(text = element_text(size = 40), legend.position = "bottom",
        legend.box = "vertical", plot.caption = element_text(hjust = 0),
        axis.text.x = element_text(size=20),
        axis.text.y = element_blank(),
        legend.text = element_text(size = 40),
        plot.margin=unit(c(2,0,0.5,-1), 'cm')) +
  scale_y_continuous(breaks = seq(from = 0.5, to = 0.8, by = 0.1), limits = c(0.45, 0.8)) +
  geom_hline(yintercept=AUCreal, alpha = 0.8, color = "#FF6633", size=2)  + # #990066
  panel_border()



plot_grid(
  gCvine, gCT, gPB,
  labels = c("CVINE", "CTGAN  TVAE", "PRIVBAYES"),
  label_size = 40,
  hjust = 0,
  vjust = 1.3,
  ncol = 3,
  rel_widths = c(13, 4, 5),
  label_x = c(0.48, 0.11, 0.22),  # Adjust these values to change label positions
  label_y = 1
)

name <- paste0("utility_realsupport2_small_CvinePrivBayesCTGANTVAE_randForest_AUC_icml", ".png", sep = "")
ggsave(filename = name, width = 25, height = 5, units = "in", bg = "white", limitsize = F)
