import json

import pandas as pd
import numpy as np
import peft
import torch
import evaluate
from tabulate import tabulate
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
from peft import LoraConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
'''
import tensorflow as tf
from keras import __version__
tf.keras.__version__ = __version__
'''
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download


def tokenize_my_text(texts, tokenizer):
    my_text = texts['text']
    tokenized_inputs = tokenizer(my_text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

'''
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    accuracy = evaluate.load("accuracy")
    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}
'''


def parse_json(json_text):
    json_decoder = json.JSONDecoder()
    json_text = json_text.replace('\r\n', '')
    try:
        decoded_json = json_decoder.decode(json_text.replace('\r\n', ''))
        return 'OK'

    except json.decoder.JSONDecodeError as e:
        decoded_json = 'Error: ' + str(e.args)
        return decoded_json


def main():
    model_name = 'PlanTL-GOB-ES/roberta-large-bne-massive'
    snapshot_download(repo_id=model_name, cache_dir='./huggingface_mirror')

    df = pd.read_csv('dataset.csv', sep=';', encoding='windows-1252', index_col=None)
    # Renombramos las columnas, ya que de lo contrario puede dar errores durante el entrenamiento
    df.columns = ['text', 'label']
    labels = df['label'].tolist()
    num_labels = len(labels)

    # Las etiquetas o labels deben ser números dentro del transformer y se han de crear mapas de los ids/identificadores esperados con id2label y label2id
    labels_to_id = {}
    ids_to_labels = {}
    i = 0
    while i < len(labels):
        ids_to_labels[i] = labels[i]
        labels_to_id[labels[i]] = i
        i = i + 1

    df['label'] = ids_to_labels.keys()
    # Creamos la configuración para modificar la capa de clasificación de nuestro modelo preentrenado al de nuestro modelo
    model_new_config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, id2label=ids_to_labels, label2id=labels_to_id, cache_dir='./huggingface_mirror')

    # Obtenemos el modelo y tokenizador del modelo ya preentrenado
    roberta_model = AutoModelForSequenceClassification.from_pretrained('PlanTL-GOB-ES/roberta-large-bne-massive', cache_dir='./huggingface_mirror', local_files_only=True)#, num_labels=num_labels, id2label=ids_to_labels, label2id=labels_to_id)
    print(roberta_model)

    if roberta_model.config.num_labels != model_new_config.num_labels or roberta_model.config.id2label != model_new_config_config.id2label:
        roberta_model.classifier.out_proj.out_features=model_new_config.num_labels
        pass
    roberta_model.config = model_new_config

    print(roberta_model)

    roberta_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./huggingface_mirror', local_files_only=True, from_pt=True)

    # Cargamos el dataset y separamos en distintas variables los valores de entrenamiento y de test

    '''
    # Comprobamos que los JSON están bien formados. De estar alguno mal formado lo indicamos
    df['Errores'] = df['JSON'].apply(parse_json)

    incorrect_dfs = df.loc[df['Errores'].str.startswith('Error: ')]
    if len(incorrect_dfs) > 0:
        print('Se han detectado errores en las siguientes filas:')

        print(tabulate(incorrect_dfs, tablefmt='grid', maxcolwidths=40, headers=df.columns))

    exit(0)
    '''
    dataset = Dataset.from_pandas(df, split='train')
    dataset = dataset.train_test_split(test_size=0.5, shuffle=True, seed=42)

    if roberta_tokenizer.pad_token is None:
        roberta_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        roberta_model.resize_token_embeddings(len(roberta_tokenizer))

    tokenized_dataset = dataset.map(tokenize_my_text, batched=True, fn_kwargs={'tokenizer': roberta_tokenizer})

    data_collator = DataCollatorWithPadding(tokenizer=roberta_tokenizer, padding=True)
    #accuracy = evaluate.load("accuracy")

    peft_config = LoraConfig(task_type='TOKEN_CLS', r=4, lora_alpha=32, lora_dropout=0.01, target_modules=['query'])
    peft_model = peft.get_peft_model(model=roberta_model, peft_config=peft_config)

    peft_model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=model_name + '-custom-lora',
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        optim='adamw_torch',
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=roberta_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    l=0
    peft_model.to('cpu')
    peft_model.eval()
    l = 0
    l = 0


if __name__ == '__main__':
    main()

