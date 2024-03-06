# install R packages
Rscript install_packages.R

# preprocess raw support2 data
Rscript preprocess_raw_support2.R

# run attribute inference attack
./AIA.sh

# save AIA and MIA results


# generate synthetic data from competitor models for utility
python synthdata_competitors_utility.py

# run executable for plots