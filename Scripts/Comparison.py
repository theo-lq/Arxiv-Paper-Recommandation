import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import torch
import time
import csv



device = 'mps' if torch.backends.mps.is_available() else 'cpu'


class CustomTrainer(Trainer):
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(p):
    logits, y_true = p
    y_proba = 1/(1+np.exp(-logits))[:, 1]

    best_threshold = 0
    best_f1 = 0
    probabilities = set(y_proba)

    for threshold in probabilities:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    y_pred = (y_proba >= best_threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    answer = {
        'threshold': best_threshold,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    return answer


def get_dataset(tokenizer, train_size):

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128).to(device)

    max_train_index = int(train_size * df.shape[0])
    abstracts = df['abstract'].tolist()
    labels = df['interest'].tolist()
    train_texts, val_texts = abstracts[:max_train_index], abstracts[max_train_index:]
    train_labels, val_labels = labels[:max_train_index], labels[max_train_index:]
    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    validation_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    validation_dataset = validation_dataset.map(tokenize_function, batched=True)
    return train_dataset, validation_dataset, train_labels



def training(model_name, train_size=0.75, class_weight=True, epoch=10, learning_rate=1e-6):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    train_dataset, validation_dataset, train_labels = get_dataset(tokenizer=tokenizer, train_size=train_size)

    training_arguments = TrainingArguments(
        output_dir='./results',
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        num_train_epochs=epoch,
        weight_decay=0.05,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],
        use_mps_device=True
    )

    if class_weight:
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).clone().detach().to(device)
        trainer = CustomTrainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            compute_metrics=compute_metrics,
            class_weights=class_weights
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            compute_metrics=compute_metrics
        )


    trainer.train()
    evaluation = trainer.evaluate()
    return evaluation


df = pd.read_csv("Dataset.csv")
df['submission_date'] = pd.to_datetime(df['submission_date'])
df = df.sort_values('submission_date')




model_names = ["answerdotai/ModernBERT-base", 'distilbert/distilbert-base-uncased', 'albert/albert-base-v2', "FacebookAI/roberta-base"]
custom_trainer_choices = [True, False]
learning_rate = 5e-6

with open('evaluation_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['Technical name', 'With custom trainer', 'Threshold', 'Accuracy', 'F1', 'Precision', 'Recall', 'Time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for model_name in model_names:
        for class_weight in [True, False]:
            print(f"{model_name} and custom trainer: {class_weight}")
            start_time = time.time()
            evaluation = training(model_name, class_weight=class_weight, epoch=10, learning_rate=learning_rate)
            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60

            row = {
                'Technical name': model_name,
                'With custom trainer': class_weight,
                'Threshold': evaluation.get('eval_threshold', ''),
                'Accuracy': evaluation.get('eval_accuracy', ''),
                'F1': evaluation.get('eval_f1', ''),
                'Precision': evaluation.get('eval_precision', ''),
                'Recall': evaluation.get('eval_recall', ''),
                'Time': f"{elapsed_time:.2f} minutes"
            }

            writer.writerow(row)
            print()


