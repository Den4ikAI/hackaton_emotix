import os
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from transformers import ViTForImageClassification, ViTConfig, TrainingArguments, Trainer
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

class VideoFramesDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        self.video_ids = []
        
        for label, class_name in enumerate(['Stress', 'Normal']):
            class_dir = os.path.join(data_dir, class_name)
            for video_dir in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_dir)
                if os.path.isdir(video_path):
                    frames = [os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.jpg')]
                    self.samples.extend(frames)
                    self.labels.extend([label] * len(frames))
                    self.video_ids.extend([video_dir] * len(frames))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return {"pixel_values": images, "labels": labels}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

# Parameters
data_dir = "rlddi"
model_name = "google/vit-base-patch16-224"
output_dir = "./vit_finetuned2"
num_labels = 2
batch_size = 64
num_epochs = 3
learning_rate = 1e-7

# Data preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = VideoFramesDataset(data_dir, transform=transform)

# Split data by video_ids
unique_video_ids = list(set(dataset.video_ids))
train_video_ids, val_video_ids = train_test_split(unique_video_ids, test_size=0.2, random_state=42)

train_indices = [i for i, vid in enumerate(dataset.video_ids) if vid in train_video_ids]
val_indices = [i for i, vid in enumerate(dataset.video_ids) if vid in val_video_ids]

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# Load model
config = ViTConfig.from_pretrained(model_name, num_labels=num_labels)
model = ViTForImageClassification.from_pretrained(model_name, ignore_mismatched_sizes=True, config=config)

# Training settings
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=500,
    weight_decay=0.02,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=200,
    save_steps=200,
    load_best_model_at_end=True,
    learning_rate=learning_rate,
    metric_for_best_model="f1",
    lr_scheduler_type="constant"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)

# Save the model
trainer.save_model(output_dir)

