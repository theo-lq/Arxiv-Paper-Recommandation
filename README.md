# Arxiv-Paper-Recommandation

With an increasing number of paper published on Arxiv, it is hard to keep up with cutting edge research. So with a personnal manually labellize dataset, we train a classifier based on BERT variants to decide whether a paper will, or won't, be interesting to me. 

The repository is organized as follow :
* **Comparison.csv** : dataset of several models trained on the same datasets with different hyperparameters (weight decay, learning rate and custom or not trainer)
* **Dataset.csv** : Arxiv paper dataset with labels
* **Finetune.ipynb** : Notebook to explain how to fine-tune the ModernBERT model. Can be used to help understand how to fine-tune any model based on HuggingFace
* **Scripts** : folder of the Python scripts that were, at least once, useful for the project
  
