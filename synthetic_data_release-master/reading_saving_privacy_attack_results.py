### reading results of AIA from server and saving to csv file:
import os
os.chdir("./synthetic_data_release-master")
from utils.analyse_results import *

directory = "./"

for i in [1]+ list(range(5, 21, 5))+ [26]:
    results = load_results_inference(dirname = f"{directory}inference_realsupport2_small_totcstOutlier_trunc{i}", dpath="./data/real_support2_small")
    results.to_csv(f"{directory}results_inference_realsupport2_small_totcst_50_0126_trunc{i}_MSE.csv", index=False)
