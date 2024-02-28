#----------------------------------
# generate data for utility:
import os 
os.chdir('./synthetic_data_release-master')
from utils.datagen import load_local_data_as_df
import numpy as np
from generative_models.data_synthesiser import Cvine, PrivBayes
from generative_models.pate_gan import PATEGAN
from generative_models.ctganSDV import CTGAN
from generative_models.tvae import TVAE
import random
import matplotlib.pyplot as plt

rawPop, metadata = load_local_data_as_df('./data/real_data_I_d20')

# setting seed
random.seed(123)
PG1 = PATEGAN(metadata=metadata, eps=1, delta=1e-5, infer_ranges=True, num_teachers=1000, n_iters=100, batch_size=64, learning_rate=1e-4, multiprocess=False) #n_iters=100
PB1 = PrivBayes(metadata=metadata, histogram_bins=25, degree=1, epsilon=1)

PB1.fit(rawPop)
synth_data_PrivBayes1 = PB1.generate_samples(50*1000)

PG1.fit(rawPop)
synth_data_PateGan1 = PG1.generate_samples(50*1000)

PB01 = PrivBayes(metadata=metadata, histogram_bins=25, degree=1, epsilon=0.1)
PG01 = PATEGAN(metadata=metadata, eps=0.1, delta=1e-5, infer_ranges=True, num_teachers=1000, n_iters=100, batch_size=64, learning_rate=1e-4, multiprocess=False)

PB01.fit(rawPop)
synth_data_PrivBayes01 = PB01.generate_samples(50*1000)

PG01.fit(rawPop)
synth_data_PateGan01 = PG01.generate_samples(50*1000)

random.seed(456)
PB10 = PrivBayes(metadata=metadata, histogram_bins=25, degree=1, epsilon=10)
PG10 = PATEGAN(metadata=metadata, eps=10, delta=1e-5, infer_ranges=True, num_teachers=1000, n_iters=100, batch_size=64, learning_rate=1e-4, multiprocess=False)

PB10.fit(rawPop)
synth_data_PrivBayes10 = PB10.generate_samples(50*1000)

PG10.fit(rawPop)
synth_data_PateGan10 = PG10.generate_samples(50*1000)


random.seed(789)
PB5 = PrivBayes(metadata=metadata, histogram_bins=25, degree=1, epsilon=5)
PG5 = PATEGAN(metadata=metadata, eps=5, delta=1e-5, infer_ranges=True, num_teachers=1000, n_iters=100, batch_size=64, learning_rate=1e-4, multiprocess=False)

PB5.fit(rawPop)
synth_data_PrivBayes5 = PB5.generate_samples(50*1000)

PG5.fit(rawPop)
synth_data_PateGan5 = PG5.generate_samples(50*1000)


synth_data_PrivBayes1.to_csv("synth_data_PrivBayes1_realI_d20.csv", index=False)
synth_data_PrivBayes01.to_csv("synth_data_PrivBayes01_realI_d20.csv", index=False)
synth_data_PateGan1.to_csv("synth_data_PateGan1_realI_d20.csv", index=False)
synth_data_PateGan01.to_csv("synth_data_PateGan01_realI_d20.csv", index=False)

synth_data_PateGan10.to_csv("synth_data_PateGan10_realI_d20.csv", index=False)
synth_data_PrivBayes10.to_csv("synth_data_PrivBayes10_realI_d20.csv", index=False)

synth_data_PateGan5.to_csv("synth_data_PateGan5_realI_d20.csv", index=False)
synth_data_PrivBayes5.to_csv("synth_data_PrivBayes5_realI_d20.csv", index=False)

# CTGAN
random.seed(178)
ctgan = CTGAN(metadata=metadata)
ctgan.fit(rawPop)
synth_data_ctgan = ctgan.generate_samples(50*1000)

synth_data_ctgan.to_csv("synth_data_CTGAN_realI_d20.csv", index=False)


# TVAE
random.seed(178)
tvae = TVAE(metadata=metadata)
tvae.fit(rawPop)
synth_data_tvae = tvae.generate_samples(50*1000)

synth_data_tvae.to_csv("synth_data_TVAE_realI_d20.csv", index=False)