import pandas as pd

datasets = ["/Users/lionellopes-quintas/Desktop/ArxivBase/Dataset.csv", "/Users/lionellopes-quintas/Desktop/ArxivBase/Arxiv_labeled_2025-24-06_23h24.csv"]

datasets = [pd.read_csv(filepath) for filepath in datasets]
dataset = pd.concat(datasets)
dataset["paper_id"] = dataset["paper_id"].astype(str)
dataset["paper_id"] = dataset["paper_id"].apply(lambda id: id[:-2] if id[-2:][0] == "v" else id)
dataset = dataset.drop_duplicates().reset_index(drop="True")
dataset.to_csv("/Users/lionellopes-quintas/Desktop/ArxivBase/Dataset.csv", index=False)