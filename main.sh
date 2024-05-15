# install R packages
Rscript install_packages.R

# preprocess raw support2 data
Rscript preprocess_raw_support2.R

# find order O^* of covariates
Rscript find_covariate_order.R

# run attribute inference attack
./AIA.sh

# Check the exit code of the script
if [ $? -eq 0 ]; then
    echo "All tmux windows have finished execution. Proceeding with other commands..."

    # save AIA results
    python read_save_attack_results.py

    # generate synthetic data from competitor models for utility
    python synthdata_competitors_utility.py

    # generate simulated real data example
    Rscript simulate_real_data.R

else
    echo "Error: Some tmux windows failed to execute."
    exit 1
fi