library(readr)
library(dplyr)
library(corrplot)
library(ggplot2)
library(data.table)
library(rvinecopulib)
library(svglite)


real_data <- read.csv("./data/preprocessed/real_support2_small.csv")


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

primary <- colnames(real_data)

# create ORDERED factors:
real_data[ , (factors) := lapply(.SD, factor, ordered = T), .SDcols = factors]
real_data[ , (numerics) := lapply(.SD, as.numeric), .SDcols = numerics]

# transforming the data to the unit cube
interim_data <- copy(real_data)
u_real <- copy(real_data)

u_real[, (numerics) := pseudo_obs(.SD), .SDcols = numerics]
u_real[, (factors) := lapply(X = .SD, FUN = rank, ties.method = "max"), .SDcols = factors]

custom_rank <- function(x){ rank(x, ties.method = "min") - 1}

for (col_name in factors) {
  new_col_name <- paste0(col_name, "_1")
  u_real[, (new_col_name) := custom_rank(interim_data[, get(col_name)])]
}

not_numerics <- setdiff(names(u_real), numerics)

u_real[, (not_numerics) := lapply(.SD, function(x) x/(n + 1)), .SDcols = not_numerics]

# remove col_names for fitting of vine
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
  
  name <- paste0("./figures/estKendallstau_synthdata_Cvine_trunc", trunc_levels[t], "_support2Small.svg", sep = "")
  # png(name)
  # corrplot(cor(synth_data[, ..numerics], method = "kendall"), method="color")
  # dev.off()
  
  cor_matrix <- cor(synth_data[, ..numerics], method = "kendall")
  cor_matrix <- apply(cor_matrix, 2, rev)
  g <-ggcorrplot(t(cor_matrix), outline.col = "white")
  ggsave(name, height=8)
  
}


name <- paste0("./figures/estKendallstau_realsupport2Small.svg", sep = "")
# png(name)
# corrplot(cor(real_data[, ..numerics], method = "kendall"), method="color")
# dev.off()

cor_matrix <- cor(real_data[, ..numerics], method = "kendall")
cor_matrix <- apply(cor_matrix, 2, rev)
g <-ggcorrplot(t(cor_matrix), outline.col = "white")
ggsave(name, height=8)

