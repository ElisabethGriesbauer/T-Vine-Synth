The following scripts are changed:

- attack_models\reconstruction.privacy: changes in class LinRegAttack
- generative_models\data_synthesiser.py: added classes Rvine, Cvine and Rvinestar1 for T-Vine-Synth
- utils\analyse_results.py: changed load_results_inference
- utils\evaluation_framework.py: commented out a line in get_probs_correct, added standardize_before_AIA
- linkage_cli.py: changed to also include vine copula synthetic data generators for T-Vine-Synth
- inference_cli.py: changed to also include vine copula synthetic data generators for T-Vine-Synth; adding MSE and standardizing features before AIA
- requirements.txt: added packages needed for T-Vine-Synth


The following scripts were added, e.g. for parameter tuning:

- generative_models\ctganSDV.py: CTGAN implementation used
- generative_models\tuning_CTGAN.py: for tuning CTGAN on simulated real data and support2 data
- generative_models\TVAE_tuning.py: for tuning TVAE on simulated real data and support2 data
- generative_models\tvae.py: wrapper around SDV's TVAE implementation
- generate_data_utility_support2.py: for generating synthetic data from competitor models used in utility analysis on support2
- generating_data_utility_realI_d20.py: for generating synthetic data from competitor models used in utility analysis on simulated real data
- conda_env.yml: conda environment used

