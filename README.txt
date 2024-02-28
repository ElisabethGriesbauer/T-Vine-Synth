Description of the files:

- evaluate_synth_data_testdata.R: see description in executable.Rmd
- executable.Rmd: file to generate and evaluate synthetic data with T-Vine-Synth on support2 data
- generate_synth_data.R:  see description in executable.Rmd
- folders synthetic_data_release-master\inference_realsupport2_small_totcstOutlier_trunc*: raw results of AIA on support2 data wrt totcst as sensitive covariate and C-vine (T-vine-synth) for different truncation levels
- making_tables.R:  see description in executable.Rmd
- preprocessing_SUPPORT2_corr_per_trunc.R: pre-processing raw support2 data and plotting estimated correlation of synthetic data generated from T-vine-synth for different truncation levels
- raw_support2.csv: raw support2 data as downloaded
- real_data_I_d20_test.csv: simulated real test data, hold out data set
- real_data_I_d20.csv: simulated real data
- real_support2_small.csv: processed support2 data used for evaluation
- synthetic_data_release-master\results_inference_realsupport2_small_totcst_50_0126_trunc*_MSE.csv: results from AIA read into python and saved to csv using function synthetic_data_release-master\reading_saving_privacy_attack_results.py
- simulated_real_data_d20.R: script to simulate real data
- simulated_real_data_estimated_corr_synth_Cvine.R: plotting estimated correlation from synthetic data generated with T-vine-synth for different truncation levels from simulated real data
- synthetic_data_release-master: attack code by Stadler et. Al.
- test_support2_small.csv: hold out test part of support2 data
- utility_realsupport2_small_Cvine_randForest_p50_testdata.RData: utility results that can also be obtained by letting executable.Rmd run (which takes some time)


For running an AIA execute:
time python inference_cli.py -D ./data/real_support2_small -RC ./tests/inference/runconfig_totcst_50_0126_trunc1.json -O ./tests/inference_realsupport2_small_totcstOutlier_trunc1 -C 0

For running an MIA execute:
time python linkage_cli.py -D ./data/real_support2_small -RC ./tests/linkage/runconfig_totcst_outliers_trunc1.json -O ./tests/linkage_realsupport2_small_totcstOutlier_trunc1