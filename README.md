# Arxiv-Paper-Recommandation

The volume of papers published on Arxiv is growing rapidly, making it challenging to stay updated with the latest research. This project aims to address this issue by training a classifier to recommend papers based on a manually labeled dataset. The classifier is built using various BERT variants (Distilled BERT, RoBERTa, ALBERT, ModernBERT) : as expected RoBERTa won !

## Repository Structure
* **Comparison.csv**: Contains a dataset of different models trained on the same dataset with varying hyperparameters (weight decay, learning rate, and custom vs. default trainer).
* **Dataset.csv**: The manually labeled dataset of Arxiv papers.
* **Finetune.ipynb**: A Jupyter notebook that explains how to fine-tune the ModernBERT model. It can also serve as a guide for fine-tuning any HuggingFace model.
* **Scripts**: A folder containing Python scripts that were useful at some point during the project.
