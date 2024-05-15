library(readr)
library(dplyr)
library(rvinecopulib)
library(data.table)
library(jsonlite)



## Helper functions

# write data to csv
write_s_r_data <- function(s_data, dir, name){
  s_data$Y <- s_data$Y %>% as.numeric() %>% as.integer()
  write.csv(s_data, paste(name, ".csv", sep = ""), row.names = F)
}


# create json file
generate_column_info <- function(df, types) {
  column_info <- list()
  
  for (i in seq_along(names(df))) {
    col_name <- names(df)[i]
    col_type <- types[i]
    
    info <- list(
      name = col_name,
      type = col_type # toupper(col_type)
    )
    
    if (col_type %in% c("Categorical", "Ordinal")) {
      levels <- levels(df[[i]])
      levels <- levels[!is.na(levels)]
      info$size <- length(levels)
      info$i2s <- as.character(levels)
    } else if (col_type == "Integer") {
      info$min <- min(df[[i]], na.rm = TRUE)
      info$max <- max(df[[i]], na.rm = TRUE)
    } else if (col_type == "Float") {
      info$min <- min(df[[i]], na.rm = TRUE)
      info$max <- max(df[[i]], na.rm = TRUE)
    }
    
    column_info[[i]] <- info
  }
  
  return(column_info)
}



## Reading data

real_data <- fread("./data/preprocessed/real_support2_small.csv")
test_data <- fread("./data/preprocessed/test_support2_small.csv")

d <- dim(real_data)[2]


## Defining parameters

# We use Kendall's $\tau$ as pairwise association measure and define the following:

# sensitive attribute
sa <- c("totcst", "crea")

# threshold on absolute pairwise association
rho <- 0.6

# initial order of the covariates
O_0 <- colnames(real_data)


# specify response name (usually "Y")
response <- colnames(real_data)[d]


## Algorithm 1

algo1 <- function(real_data, sa, response, rho, O_0){
  
    d <- dim(real_data)[2]
    
    # initialize O_star and set response on last index
    O_star <- c(rep(NA, d-1), response)
    
    # compute pairwise association measure between sensitive attribute and remaining features
    pa <- cor(real_data[, ..sa], real_data[, -..sa][, -..response], method = "kendall") %>% 
      abs() 
    
    # define set K
    K <- colnames(pa)[apply(pa > rho, 2, any)]
    
    # order rhos
    pa <- data.table(pa)
    above_threshold <- pa[, apply(.SD, 2, max)][eval(K)]
    K_ordered <- names(above_threshold)[order(above_threshold, decreasing = TRUE)]
    
    # place sensitive features on first positions
    O_star[1:length(sa)] <- sa
    
    # place features with abs pairwise association above rho next
    O_star[length(sa) + (1:length(above_threshold))] <- K_ordered
    
    # place remaining features
    O_star[is.na(O_star)] <- O_0[!O_0 %in% c(names(above_threshold), eval(response), eval(sa))]
    
    return(O_star)
  
}


# finding order
O_star <- algo1(real_data, sa, response, rho, O_0)



## Saving re-ordered data for evaluation

# reorder train data
setcolorder(real_data, O_star)

# reorder test data
setcolorder(test_data, O_star)

# save reordered data
write_s_r_data(real_data %>% as.data.frame(), name = "./data/preprocessed/real_support2_small")
write_s_r_data(test_data %>% as.data.frame(), name = "./data/preprocessed/test_support2_small")


# create data.json file for AIA and MIA
factors <- colnames(real_data)[d]
real_data[ , (factors) := lapply(.SD, factor, ordered = T), .SDcols = factors]
col_type <- c(rep("Float", d-1), "Categorical")

json_output <- toJSON(list(columns = generate_column_info(real_data, col_type)), pretty = TRUE, auto_unbox = TRUE)
writeLines(json_output, "./data/preprocessed/real_support2_small.json")
