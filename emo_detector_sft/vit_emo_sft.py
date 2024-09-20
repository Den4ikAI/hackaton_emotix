import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop, Compose, Normalize, RandomRotation, RandomResizedCrop,
    RandomHorizontalFlip, RandomAdjustSharpness, Resize, ToTensor
)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from imblearn.over_sampling import RandomOverSampler
import accelerate
import evaluate
from datasets import Dataset, Image, ClassLabel
from transformers import (
    TrainingArguments, Trainer, ViTImageProcessor,
    ViTForImageClassification, DefaultDataCollator
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def load_data():
    file_names = []
    labels = []
    for file in sorted((Path('facial-emotion-expressions/images/images/').glob('*/*/*.*'))):
        label = str(file).split('/')[-2]
        labels.append(label)
        file_names.append(str(file))
    
    df = pd.DataFrame.from_dict({"image": file_names, "label": labels})
    return df

def oversample_data(df):
    y = df[['label']]
    df = df.drop(['label'], axis=1)
    ros = RandomOverSampler(random_state=83)
    df, y_resampled = ros.fit_resample(df, y)
    df['label'] = y_resampled
    return df

def prepare_dataset(df):
    dataset = Dataset.from_pandas(df).cast_column("image", Image())
    labels_list = ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy']
    label2id = {label: i for i, label in enumerate(labels_list)}
    id2label = {i: label for label, i in label2id.items()}
    
    ClassLabels = ClassLabel(num_classes=len(labels_list), names=labels_list)
    
    def map_label2id(example):
        example['label'] = ClassLabels.str2int(example['label'])
        return example
    
    dataset = dataset.map(map_label2id, batched=True)
    dataset = dataset.cast_column('label', ClassLabels)
    dataset = dataset.train_test_split(test_size=0.4, shuffle=True, stratify_by_column="label")
    return dataset, label2id, id2label

def create_transforms(processor):
    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]
    normalize = Normalize(mean=image_mean, std=image_std)
    
    train_transforms = Compose([
        Resize((size, size)), RandomRotation(90), RandomAdjustSharpness(2),
        RandomHorizontalFlip(0.5), ToTensor(), normalize
    ])
    
    val_transforms = Compose([
        Resize((size, size)), ToTensor(), normalize
    ])
    
    return train_transforms, val_transforms

def apply_transforms(examples, transforms):
    examples['pixel_values'] = [transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example['label'] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids
    predicted_labels = predictions.argmax(axis=1)
    acc_score = accuracy_score(label_ids, predicted_labels)
    return {"accuracy": acc_score}

def train_model(train_data, test_data, model, processor, num_labels):
    model_name = "vit_emo_sft"
    num_train_epochs = 5
    
    args = TrainingArguments(
        output_dir=model_name,
        logging_dir='./logs',
        evaluation_strategy="epoch",
        learning_rate=1e-7,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=8,
        num_train_epochs=num_train_epochs,
        weight_decay=0.02,
        warmup_steps=50,
        remove_unused_columns=False,
        save_strategy='epoch',
        load_best_model_at_end=True,
        save_total_limit=1,
        report_to="none"
    )
    
    trainer = Trainer(
        model,
        args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )
    
    trainer.train()
    return trainer



df = load_data()
df = oversample_data(df)
dataset, label2id, id2label = prepare_dataset(df)

model_str = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_str)
train_transforms, val_transforms = create_transforms(processor)

train_data = dataset['train'].set_transform(lambda examples: apply_transforms(examples, train_transforms))
test_data = dataset['test'].set_transform(lambda examples: apply_transforms(examples, val_transforms))

model = ViTForImageClassification.from_pretrained(model_str, num_labels=len(label2id))
model.config.id2label = id2label
model.config.label2id = label2id

trainer = train_model(train_data, test_data, model, processor, len(label2id))
outputs = trainer.predict(test_data)
y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

labels_list = list(id2label.values())
print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=labels_list, digits=4))
trainer.save_model()
