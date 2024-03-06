from synthetic_data_release.utils.analyse_results import load_results_inference

directory = "./output/"

for i in [1]+ list(range(5, 21, 5))+ [26]:
    for sa in ["totcst"]:
        results = load_results_inference(dirname = f"{directory}inference_realsupport2_small_{sa}Outlier_trunc{i}", dpath="./synthetic_data_release/data/real_support2_small")
        results.to_csv(f"{directory}results_inference_realsupport2_small_{sa}_50_0126_trunc{i}_MSE.csv", index=False)
