import os
import sys
import glob
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms, datasets
from tqdm import tqdm
import argparse

from emotixV4 import EmotixResNet
class ImbalancedDatasetSampler(data.sampler.Sampler):
    def __init__(self, dataset, indices: list = None, num_samples: int = None):
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

        # self.weights = self.weights.clamp(min=1e-5)

    def _get_labels(self, dataset):
        if isinstance(dataset, datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torch.utils.data.Subset):
            return [dataset.dataset.imgs[i][1] for i in dataset.indices]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
    

class EmotionDatasetLoader:
    def __init__(self, dataset_path: str, num_classes: int = 8):
        self.dataset_path = dataset_path
        self.num_classes = num_classes
        
    def create_dataset(self, phase: str, transform: Optional[transforms.Compose] = None) -> data.Dataset:
        dataset = datasets.ImageFolder(os.path.join(self.dataset_path, phase), transform=transform)
        
        if self.num_classes == 7:
            filtered_indices = [i for i in range(len(dataset)) if dataset.imgs[i][1] != 7]
            dataset = data.Subset(dataset, filtered_indices)
        
        return dataset

class LossFunctions:
    @staticmethod
    def affinity_loss(device: torch.device, num_class: int = 8, feat_dim: int = 512) -> nn.Module:
        class AffinityLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_class = num_class
                self.feat_dim = feat_dim
                self.gap = nn.AdaptiveAvgPool2d(1)
                self.centers = nn.Parameter(torch.randn(num_class, feat_dim).to(device))

            def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                x = self.gap(x).view(x.size(0), -1)
                batch_size = x.size(0)
                
                distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_class) + \
                          torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_class, batch_size).t()
                distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

                classes = torch.arange(self.num_class).long().to(device)
                labels = labels.unsqueeze(1).expand(batch_size, self.num_class)
                mask = labels.eq(classes.expand(batch_size, self.num_class))

                dist = distmat * mask.float()
                dist = dist / self.centers.var(dim=0).sum()

                return dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return AffinityLoss()

    @staticmethod
    def partition_loss() -> nn.Module:
        class PartitionLoss(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                num_head = x.size(1)
                return torch.log(1 + num_head / (x.var(dim=1).mean() + sys.float_info.epsilon)) if num_head > 1 else 0

        return PartitionLoss()

class TrainingManager:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._configure_cuda()
        
    def _configure_cuda(self):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enabled = True
    
    def _create_data_transforms(self, is_train: bool = True) -> transforms.Compose:
        base_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        if is_train:
            train_transforms = base_transforms.copy()
            train_transforms.insert(1, transforms.RandomHorizontalFlip())
            train_transforms.insert(2, transforms.RandomApply([
                transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2))
            ], p=0.7))
            train_transforms.append(transforms.RandomErasing())
            return transforms.Compose(train_transforms)
        
        return transforms.Compose(base_transforms)

    def train(self):
        dataset_loader = EmotionDatasetLoader(self.args.aff_path, self.args.num_class)
        
        train_dataset = dataset_loader.create_dataset('train', self._create_data_transforms(is_train=True))
        val_dataset = dataset_loader.create_dataset('val', self._create_data_transforms(is_train=False))
        
        train_loader = self._create_data_loader(train_dataset, is_train=True)
        val_loader = self._create_data_loader(val_dataset, is_train=False)
        
        model, optimizer, scheduler = self._initialize_training_components(train_dataset)
        
        self._run_training_loop(model, train_loader, val_loader, optimizer, scheduler)

    def _create_data_loader(self, dataset, is_train: bool = True):
        sampler = ImbalancedDatasetSampler(dataset) if is_train else None
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            sampler=sampler,
            shuffle=False if is_train else False,
            pin_memory=True
        )

    def _initialize_training_components(self, train_dataset):
        model = EmotixResNet().to(self.device)
        
        criterion_cls = torch.nn.CrossEntropyLoss().to(self.device)
        criterion_af = LossFunctions.affinity_loss(self.device, num_class=self.args.num_class)
        criterion_pt = LossFunctions.partition_loss()
        
        params = list(model.parameters()) + list(criterion_af.parameters())
        optimizer = torch.optim.Adam(params, self.args.lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6)
        
        return model, optimizer, scheduler

    def _run_training_loop(self, model, train_loader, val_loader, optimizer, scheduler):
        best_acc = 0
        for epoch in tqdm(range(1, self.args.epochs + 1)):
            train_acc, train_loss = self._train_epoch(model, train_loader, optimizer)
            val_acc, val_loss = self._validate_epoch(model, val_loader, scheduler)
            
            tqdm.write(f'[Epoch {epoch}] Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.3f}')
            tqdm.write(f'[Epoch {epoch}] Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.3f}')
            
            best_acc = max(val_acc, best_acc)
            tqdm.write(f"Best Accuracy: {best_acc}")
            
            self._save_model_checkpoint(model, optimizer, epoch, val_acc)

    def _train_epoch(self, model, train_loader, optimizer):
        model.train()
        running_loss, correct_sum, iter_cnt = 0.0, 0, 0
        
        for imgs, targets in train_loader:
            iter_cnt += 1
            optimizer.zero_grad()

            imgs, targets = imgs.to(self.device), targets.to(self.device)
            out, feat, heads = model(imgs)

            criterion_cls = torch.nn.CrossEntropyLoss().to(self.device)
            criterion_af = LossFunctions.affinity_loss(self.device, num_class=self.args.num_class)
            criterion_pt = LossFunctions.partition_loss()
            
            loss = criterion_cls(out, targets) + criterion_af(feat, targets) + criterion_pt(heads)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss
            _, predicts = torch.max(out, 1)
            correct_sum += torch.eq(predicts, targets).sum()

        acc = correct_sum.float() / len(train_loader.dataset)
        return acc.item(), (running_loss / iter_cnt).item()

    def _validate_epoch(self, model, val_loader, scheduler):
        model.eval()
        running_loss, iter_cnt, bingo_cnt, sample_cnt = 0.0, 0, 0, 0
        
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(self.device), targets.to(self.device)
                out, feat, heads = model(imgs)

                criterion_cls = torch.nn.CrossEntropyLoss().to(self.device)
                criterion_af = LossFunctions.affinity_loss(self.device, num_class=self.args.num_class)
                criterion_pt = LossFunctions.partition_loss()
                
                loss = criterion_cls(out, targets) + criterion_af(feat, targets) + criterion_pt(heads)
                
                running_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(out, 1)
                bingo_cnt += torch.eq(predicts, targets).sum().cpu()
                sample_cnt += out.size(0)

            scheduler.step()
            acc = bingo_cnt.float() / sample_cnt
            return acc.item(), (running_loss / iter_cnt).item()

    def _save_model_checkpoint(self, model, optimizer, epoch, acc):
        save_conditions = {
            7: acc > 0.65,
            8: acc > 0.62
        }
        
        if save_conditions.get(self.args.num_class, False):
            checkpoint_path = os.path.join(
                'checkpoints', 
                f'affecnet{self.args.num_class}_epoch{epoch}_acc{acc:.4f}.pth'
            )
            torch.save({
                'iter': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            tqdm.write('Model saved.')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aff_path', type=str, help='dataset path.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for adam.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=4, help='Number of attention head.')
    parser.add_argument('--num_class', type=int, default=8, help='Number of class.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    training_manager = TrainingManager(args)
    training_manager.train()

if __name__ == "__main__":
    main()