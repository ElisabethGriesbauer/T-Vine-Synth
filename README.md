# T-Vine-Synth: A Truncated C-Vine Copula Generator of Synthetic Tabular Data to Balance Privacy and Utility

 This repository accompanies the paper "T-Vine-Synth: A Truncated C-Vine Copula Generator of Synthetic Tabular Data to Balance Privacy and Utility".


### Usage
- Software used and needed: Python 3.10.13, R version 4.3.1 (2023-06-16) -- "Beagle Scouts", tmux 3.3a, conda 23.10.0
- Set up a conda env `attack_env` from the `attack_env.yml` file  for the privacy attacks and the conda env `privpgd` from the `privpgd.yml` file for the PrivPGD model. Activate `attack_env`.
- Execute `main.sh` for installing necessary R packages, data preprocessing, privacy evaluation on vines and saving privacy results as .csv.
- Run `executable.Rmd` to obtain utility evaluation and privacy, utility and privacy-utility plots.
- Execute `corr_plots_simulated_real_data.R` and `corr_plots_support2.R` to obtain correlation plots on real and synthetic data for simulated and real-world data.

### Cite
@article{griesbauer2025tvinesynth,
  title={TVineSynth: A Truncated C-Vine Copula Generator of Synthetic Tabular Data to Balance Privacy and Utility},
  author={Griesbauer, Elisabeth and Czado, Claudia and Frigessi, Arnoldo and Haff, Ingrid Hob{\ae}k},
  journal={arXiv preprint arXiv:2503.15972},
  year={2025}
}