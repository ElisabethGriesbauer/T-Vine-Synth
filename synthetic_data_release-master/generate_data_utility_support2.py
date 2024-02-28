
#----------------------------------
# generate data for utility for support2 SMALL:
import os 
os.chdir('./synthetic_data_release-master')
from utils.datagen import load_local_data_as_df
import numpy as np
from generative_models.data_synthesiser import PrivBayes
from generative_models.ctganSDV import CTGAN
from generative_models.tvae import TVAE
import random
import matplotlib.pyplot as plt

rawPop, metadata = load_local_data_as_df('./data/real_support2_small')

# setting seed
random.seed(123)
PB01 = PrivBayes(metadata=metadata, histogram_bins=25, degree=1, epsilon=0.1)
PB1 = PrivBayes(metadata=metadata, histogram_bins=25, degree=1, epsilon=1)
PB5 = PrivBayes(metadata=metadata, histogram_bins=25, degree=1, epsilon=5)

random.seed(789)
PB01.fit(rawPop)
synth_data_PrivBayes01 = PB01.generate_samples(50*rawPop.shape[0])
PB1.fit(rawPop)
synth_data_PrivBayes1 = PB1.generate_samples(50*rawPop.shape[0])
PB5.fit(rawPop)
synth_data_PrivBayes5 = PB5.generate_samples(50*rawPop.shape[0])

synth_data_PrivBayes01.to_csv("synth_data_PrivBayes01_real_support2_small.csv", index=False)
synth_data_PrivBayes1.to_csv("synth_data_PrivBayes1_real_support2_small.csv", index=False)
synth_data_PrivBayes5.to_csv("synth_data_PrivBayes5_real_support2_small.csv", index=False)


# CTGAN
random.seed(178)
ctgan = CTGAN(metadata=metadata, epochs = 400, batch_size = 100)
ctgan.fit(rawPop)
synth_data_ctgan = ctgan.generate_samples(50*rawPop.shape[0])
synth_data_ctgan.to_csv("synth_data_CTGAN_real_support2_small.csv", index=False)


# TVAE
random.seed(179)
tvae = TVAE(metadata=metadata, epochs = 800, batch_size = 100,embedding_dim = 4)
tvae.fit(rawPop)
synth_data_tvae = tvae.generate_samples(50*rawPop.shape[0])
synth_data_tvae.to_csv("synth_data_TVAE_real_support2_small.csv", index=False)

