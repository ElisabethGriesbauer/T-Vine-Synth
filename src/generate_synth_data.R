generate_synth_data <- function(real_data, p_synth = 1, attribute_names, factor_position, vine_estimation = "parametric", vine_topology = "Cvine", t_lvl = Inf){
  
  
  # function that generates synthetic data of same size from real data and returns the synthetic data set
      # real_data: real data set as data frame
      # p_synth: functions generates n * p_synth synthetic observations
      # attribute_names: vector of strings containing names of the covariates in the chosen ordering; Y needs to be on the last index, so the last entry of the attribute_names vector
      # factor_position: vector of integers indicating the column index of categorical attributes
      # vine_estimation: "par" for parametric pair copula estimation (also possible: "nonpar" for non-parametric or "mixed" pair copula estimation)
      # vine_topology: either "Cvine", "Rvine" or "star1" (for Rvine star1)
      # t_lvl: truncation level of the vine copula model
      
      
      
      d <- length(attribute_names)
      n <- dim(real_data)[1]
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
      
      
      # vine structure
      if(vine_topology == "Rvine"){
        # structure <- NA
        structure <- rvine_structure(order = 1:d)
      } else if (vine_topology == "Cvine"){
        structure <- cvine_structure(order = 1:d, trunc_lvl = t_lvl)
      } else {
        structure <- rvine_structure(order = 1:d, struct_array = list(rep(d, (d-1))), is_natural_order = T)
        
        if(t_lvl < Inf){
          structure <- truncate_model(structure, trunc_lvl = t_lvl)  
        }
        
      }
      
      v_types <- rep('c', d)
      v_types[factor_position] <- 'd'
      
      # fitting the vine copula model
      if(vine_estimation == "par"){
        vine_model <- vinecop(data = u_real, var_types = v_types, 
                              par_method = "mle", family_set = "parametric", structure, nonpar_method = "linear", trunc_lvl = t_lvl)
      } else if (vine_estimation == "nonpar"){
        vine_model <- vinecop(data = u_real, var_types = v_types, 
                              family_set = "nonparametric", nonpar_method = "linear", structure, par_method = "mle", trunc_lvl = t_lvl)
      } else {
        vine_model <- vinecop(data = u_real, var_types = v_types, 
                              family_set = "all", par_method = "mle", nonpar_method = "linear", structure, trunc_lvl = t_lvl)
      }
      
      # simulating synthetic data on the unit cube from the vine
      u_synth <- rvinecop(n * p_synth, vine_model) %>% data.table()
      names(u_synth) <- primary
  
      
      # back-transforming the data to the original scale
      ## numerics ##
      synth_data <- data.table(matrix(NA, nrow = p_synth*n, ncol = length(colnames(real_data)), dimnames = list(NULL, colnames(real_data))))
      
      for (x in numerics){
        synth_data[, (x) := quantile(real_data[[x]], probs = u_synth[[x]], type = 8)]
      }
      
      
      ## factors ##
      for (x in factors){
        synth_data[, (x) := quantile(real_data[[x]], probs = u_synth[[x]], type = 3)]
      }
      
      return(synth_data)
      
      
}





## same as generate_synth_data but this time for several truncation levels of the vine at the same time; 
## gives a list of synthetic data sets of dim (n*p_synth x d) where list[[t]] contains data with truncation level trunc_levels[t]

generate_synth_data_all_truncs <- function(real_data, p_synth = 1, attribute_names, factor_position, vine_estimation = "par", vine_topology = "Cvine", trunc_levels = c(1:(d-1))){
  # real_data: real data set as data frame
  # p_synth: functions generates n * p_synth synthetic observations
  # attribute_names: vector of strings containing names of the covariates in the chosen ordering; Y needs to be on the last index, so the last entry of the attribute_names vector
  # factor_position: vector of integers indicating the column index of categorical attributes
  # vine_estimation: "par" for parametric pair copula estimation (also possible: "nonpar" for non-parametric or "mixed" pair copula estimation)
  # vine_topology: either "Cvine", "Rvine" or "star1" (for Rvine star1)
  # trunc_levels: user specified truncation levels of the vine copula model
  
  
  
  d <- length(attribute_names)
  n <- dim(real_data)[1]
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
  
  
  # vine structure
  if(vine_topology == "Rvine"){
    # structure <- NA
    structure <- rvine_structure(order = 1:d)
  } else if (vine_topology == "Cvine"){
    structure <- cvine_structure(order = 1:d)
  } else {
    structure <- rvine_structure(order = 1:d, struct_array = list(rep(d, (d-1))), is_natural_order = T)
  }
  
  v_types <- rep('c', d)
  v_types[factor_position] <- 'd'
  
  # fitting the vine copula model
  if(vine_estimation == "par"){
    vine_model <- vinecop(data = u_real, var_types = v_types, 
                          par_method = "mle", family_set = "parametric", structure, nonpar_method = "linear")
  } else if (vine_estimation == "nonpar"){
    vine_model <- vinecop(data = u_real, var_types = v_types, 
                          family_set = "nonparametric", nonpar_method = "linear", structure, par_method = "mle")
  } else {
    vine_model <- vinecop(data = u_real, var_types = v_types, 
                          family_set = "all", par_method = "mle", nonpar_method = "linear", structure)
  }
  
  
  list_synth_data <- list()
  # loop over truncation levels; list_synth_data[[t]] contains synth_data of trunc level t
  for (t in 1:length(trunc_levels)) {
    
    v_model <- truncate_model(vine_model, trunc_lvl = trunc_levels[t])
    # simulating synthetic data on the unit cube from the vine
    u_synth <- rvinecop(n * p_synth, v_model) %>% data.table()
    names(u_synth) <- primary
    
    
    # back-transforming the data to the original scale
    ## numerics ##
    synth_data <- data.table(matrix(NA, nrow = p_synth*n, ncol = length(colnames(real_data)), dimnames = list(NULL, colnames(real_data))))
    
    for (x in numerics){
      synth_data[, (x) := quantile(real_data[[x]], probs = u_synth[[x]], type = 8)]
    }
    
    
    ## factors ##
    for (x in factors){
      synth_data[, (x) := quantile(real_data[[x]], probs = u_synth[[x]], type = 3)]
    }
    
    list_synth_data[[t]] <- synth_data
  }
  
  
  names(list_synth_data) <- paste(trunc_levels)
  return(list_synth_data)
  
  
}
