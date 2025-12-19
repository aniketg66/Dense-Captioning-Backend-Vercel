#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure PyTorch Chart Type Classification Training Script - DocFigure Dataset"""

import os
import csv
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class DocFigureDataset(Dataset):
    """DocFigure Dataset with 28 categories using PyTorch Dataset"""
    
    # DocFigure 28 categories as described in the paper
    CHART_TYPES = [
        'Line graph', 'Natural image', 'Table', '3D objects', 'Bar plot', 
        'Scatter plot', 'Medical image', 'Sketch', 'Geographic map', 'Flow chart',
        'Heat map', 'Mask', 'Block diagram', 'Venn diagram', 'Confusion matrix',
        'Histogram', 'Box plot', 'Vector plot', 'Pie chart', 'Surface plot',
        'Algorithm', 'Contour plot', 'Tree diagram', 'Bubble chart', 'Polar plot',
        'Area chart', 'Pareto chart', 'Radar chart'
    ]
    
    def __init__(self, images_dir, ann_file, transform=None, is_train=True):
        self.images_dir = images_dir
        self.transform = transform
        self.chart_type_to_label = {chart_type: idx for idx, chart_type in enumerate(self.CHART_TYPES)}
        
        # Load data
        self.data_list = self.load_data_list(ann_file)
        print(f"Loaded {len(self.data_list)} samples for {'training' if is_train else 'validation'}")
        
        # Print class distribution
        self.print_class_distribution()
    
    def load_data_list(self, ann_file):
        """Load CSV annotation file and return list of data info"""
        data_list = []
        
        if not os.path.exists(ann_file):
            print(f"Warning: Annotation file {ann_file} does not exist!")
            return data_list
        
        # Create flexible mapping for singular/plural forms and known variations
        flexible_mapping = {}
        for chart_type in self.CHART_TYPES:
            flexible_mapping[chart_type.lower()] = chart_type
            # Add plural forms
            if chart_type.lower() == 'table':
                flexible_mapping['tables'] = chart_type
            elif chart_type.lower().endswith('graph'):
                flexible_mapping[chart_type.lower() + 's'] = chart_type
            elif chart_type.lower().endswith('plot'):
                flexible_mapping[chart_type.lower() + 's'] = chart_type
            elif chart_type.lower().endswith('chart'):
                flexible_mapping[chart_type.lower() + 's'] = chart_type
            elif chart_type.lower().endswith('diagram'):
                flexible_mapping[chart_type.lower() + 's'] = chart_type
            elif chart_type.lower().endswith('map'):
                flexible_mapping[chart_type.lower() + 's'] = chart_type
            elif chart_type.lower().endswith('image'):
                flexible_mapping[chart_type.lower() + 's'] = chart_type
        
        # Add specific mappings for known variations in the dataset
        flexible_mapping.update({
            'graph plots': 'Line graph',          # "Graph plots" → "Line graph"
            'graph plot': 'Line graph',           # singular variant
            'sketches': 'Sketch',                 # "Sketches" → "Sketch"
            'line graphs': 'Line graph',          # plural of line graph
            'natural images': 'Natural image',    # plural variant
            'medical images': 'Medical image',    # plural variant
        })
        
        with open(ann_file, 'r', encoding='utf-8') as f:
            # Skip header if present, or treat first line as data
            reader = csv.reader(f)
            for line_num, row in enumerate(reader, 1):
                if len(row) < 2:
                    print(f"Warning: Skipping line {line_num} - insufficient columns: {row}")
                    continue
                
                img_filename = row[0].strip()
                category = row[1].strip()
                
                # Skip if category is not in our defined types
                if category not in self.chart_type_to_label:
                    # Try to find close match (case insensitive and flexible mapping)
                    category_lower = category.lower()
                    found_match = False
                    
                    # First try exact case-insensitive match
                    for chart_type in self.CHART_TYPES:
                        if chart_type.lower() == category_lower:
                            category = chart_type
                            found_match = True
                            break
                    
                    # Then try flexible mapping (handles plural forms)
                    if not found_match and category_lower in flexible_mapping:
                        category = flexible_mapping[category_lower]
                        found_match = True
                    
                    if not found_match:
                        print(f"Warning: Unknown category '{category}' for image {img_filename}")
                        print(f"Available categories: {list(self.chart_type_to_label.keys())}")
                        continue
                
                img_path = os.path.join(self.images_dir, img_filename)
                
                if os.path.exists(img_path):
                    label = self.chart_type_to_label[category]
                    # Validate label is in expected range
                    if label < 0 or label >= len(self.CHART_TYPES):
                        print(f"ERROR: Invalid label {label} for category '{category}' in image {img_filename}")
                        continue
                        
                    data_list.append({
                        'img_path': img_path,
                        'gt_label': label,
                        'chart_type': category,
                        'filename': img_filename
                    })
                else:
                    print(f"Warning: Image file not found: {img_path}")
        
        return data_list
    
    def print_class_distribution(self):
        """Print class distribution"""
        class_counts = {}
        for data_info in self.data_list:
            chart_type = data_info['chart_type']
            class_counts[chart_type] = class_counts.get(chart_type, 0) + 1
        
        print("Class distribution:")
        for chart_type in self.CHART_TYPES:
            count = class_counts.get(chart_type, 0)
            if count > 0:
                print(f"  {chart_type}: {count}")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_info = self.data_list[idx]
        
        # Load image with proper PIL warning fix
        try:
            img = Image.open(data_info['img_path'])
            # CRITICAL: Ensure no palettes or alpha channels sneak through
            # This fixes PIL warnings and guarantees true 3-channel images
            if img.mode != 'RGB':
                img = img.convert('RGB')
        except Exception as e:
            print(f"Error loading image {data_info['img_path']}: {e}")
            # Return a dummy RGB image
            img = Image.new('RGB', (224, 224), color='white')
        
        # Apply transforms
        if self.transform:
            image = self.transform(img)
        else:
            image = img
        
        label = data_info['gt_label']
        
        # Validate label range
        if label < 0 or label >= len(self.CHART_TYPES):
            print(f"ERROR: Invalid label {label} for image {data_info['filename']}. Chart type: {data_info['chart_type']}")
            print(f"Valid label range is [0, {len(self.CHART_TYPES)-1}]")
            # Clip to valid range
            label = max(0, min(label, len(self.CHART_TYPES)-1))
        
        return image, label

class ChartClassifier(nn.Module):
    """Enhanced Chart Type Classifier using ResNet50 with improved regularization"""
    
    def __init__(self, num_classes=28):  # 28 for DocFigure dataset
        super(ChartClassifier, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=True)
        
        # Improved dropout for better generalization
        self.dropout = nn.Dropout(0.6)  # Increased from 0.5 to 0.6
        
        # Replace the final fully connected layer with enhanced architecture
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Linear(512, num_classes)
        )
        
        print(f"Enhanced ChartClassifier initialized with {num_classes} output classes")
        print(f"Using improved dropout rate: 0.6")
        
    def forward(self, x):
        return self.backbone(x)

def create_transforms():
    """Create enhanced training and validation transforms with stronger regularization"""
    
    # Enhanced training transforms with stronger augmentation and regularization
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.8, 1.0),                           # Scale range for crop
            ratio=(3/4, 4/3),                           # Aspect ratio range
            interpolation=Image.BILINEAR                # Consistent interpolation
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),          # Increased from 5
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Stronger jitter
        transforms.RandomAffine(degrees=5, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.ToTensor(),                          # [0,1] pixel range
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # NEW: Random erasing for regularization
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Validation transforms using Resize + CenterCrop (consistent with inference)
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=Image.BILINEAR),    # Resize to 256 with BILINEAR
        transforms.CenterCrop(224),                              # Central 224×224 crop
        transforms.ToTensor(),                                   # [0,1] pixel range
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    return train_transform, val_transform

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """Enhanced training epoch with per-batch LR scheduling"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Step scheduler every batch for OneCycleLR
        if scheduler is not None:
            scheduler.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        accuracy = 100 * correct / total
        current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}', 
            'Acc': f'{accuracy:.2f}%',
            'LR': f'{current_lr:.6f}'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            accuracy = 100 * correct / total
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{accuracy:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_predictions, all_labels

def validate_dataset_labels(train_dataset, val_dataset):
    """Validate that all labels in datasets are within valid range"""
    print("Validating dataset labels...")
    
    all_data = train_dataset.data_list + val_dataset.data_list
    all_labels = [data['gt_label'] for data in all_data]
    all_chart_types = [data['chart_type'] for data in all_data]
    
    # Check for any labels outside valid range
    max_valid_label = len(DocFigureDataset.CHART_TYPES) - 1
    invalid_samples = []
    
    for i, label in enumerate(all_labels):
        if label < 0 or label > max_valid_label:
            invalid_samples.append({
                'index': i,
                'label': label,
                'chart_type': all_chart_types[i],
                'filename': all_data[i]['filename']
            })
    
    if invalid_samples:
        print(f"ERROR: Found {len(invalid_samples)} samples with invalid labels:")
        for sample in invalid_samples[:10]:  # Show first 10
            print(f"  {sample['filename']}: label={sample['label']}, type='{sample['chart_type']}'")
        if len(invalid_samples) > 10:
            print(f"  ... and {len(invalid_samples) - 10} more")
        raise ValueError(f"Dataset contains invalid labels. Valid range is [0, {max_valid_label}]")
    
    # Summary statistics
    unique_labels = sorted(set(all_labels))
    print(f"Label validation passed!")
    print(f"  Total samples: {len(all_labels)}")
    print(f"  Unique labels found: {len(unique_labels)}")
    print(f"  Label range: {min(unique_labels)} to {max(unique_labels)}")
    print(f"  Expected range: [0, {max_valid_label}]")
    
    # Check for missing labels
    missing_labels = [i for i in range(len(DocFigureDataset.CHART_TYPES)) if i not in unique_labels]
    if missing_labels:
        print(f"  Warning: {len(missing_labels)} chart types have no samples:")
        for label in missing_labels:
            print(f"    {label}: {DocFigureDataset.CHART_TYPES[label]}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='DocFigure Chart Type Classification Training')
    parser.add_argument('--images_dir', default='/content/drive/MyDrive/Research Summer 2025/Dense Captioning Toolkit/CHART-Classification/images', 
                       help='Directory containing images')
    parser.add_argument('--train_ann', default='/content/drive/MyDrive/Research Summer 2025/Dense Captioning Toolkit/CHART-Classification/annotation/train.txt', 
                       help='Training annotations CSV file')
    parser.add_argument('--val_ann', default='/content/drive/MyDrive/Research Summer 2025/Dense Captioning Toolkit/CHART-Classification/annotation/val.txt', 
                       help='Validation annotations CSV/TXT file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--work_dir', default='./work_dirs/docfigure_classifier', help='Work directory')
    parser.add_argument('--device', default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Create work directory
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create transforms
    train_transform, val_transform = create_transforms()
    
    # Create datasets
    print("Loading training dataset...")
    train_dataset = DocFigureDataset(
        images_dir=args.images_dir,
        ann_file=args.train_ann,
        transform=train_transform,
        is_train=True
    )
    
    print("\nLoading validation dataset...")
    val_dataset = DocFigureDataset(
        images_dir=args.images_dir,
        ann_file=args.val_ann,
        transform=val_transform,
        is_train=False
    )
    
    # Show detailed class distribution for both datasets
    print("\n" + "="*80)
    print("DOCFIGURE DATASET SUMMARY")
    print("="*80)
    
    # Training dataset counts
    train_counts = {}
    for data_info in train_dataset.data_list:
        chart_type = data_info['chart_type']
        train_counts[chart_type] = train_counts.get(chart_type, 0) + 1
    
    # Validation dataset counts  
    val_counts = {}
    for data_info in val_dataset.data_list:
        chart_type = data_info['chart_type']
        val_counts[chart_type] = val_counts.get(chart_type, 0) + 1
    
    # Display summary table
    all_chart_types = set(list(train_counts.keys()) + list(val_counts.keys()))
    all_chart_types = sorted(all_chart_types)
    
    print(f"{'Chart Type':<20} {'Train Count':<12} {'Val Count':<12} {'Total':<12}")
    print("-" * 70)
    
    total_train = 0
    total_val = 0
    
    for chart_type in all_chart_types:
        train_count = train_counts.get(chart_type, 0)
        val_count = val_counts.get(chart_type, 0)
        total_count = train_count + val_count
        
        print(f"{chart_type:<20} {train_count:<12} {val_count:<12} {total_count:<12}")
        total_train += train_count
        total_val += val_count
    
    print("-" * 70)
    print(f"{'TOTAL':<20} {total_train:<12} {total_val:<12} {total_train + total_val:<12}")
    print("="*80 + "\n")
    
    # Validate dataset labels before proceeding
    validate_dataset_labels(train_dataset, val_dataset)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Calculate number of classes from loaded data and validate
    all_labels = set()
    for data_info in train_dataset.data_list + val_dataset.data_list:
        all_labels.add(data_info['gt_label'])
    
    print(f"Number of classes found in data: {len(all_labels)}")
    print(f"Label range in data: {min(all_labels)} to {max(all_labels)}")
    print(f"Total chart types defined: {len(DocFigureDataset.CHART_TYPES)}")
    
    # Validate that all labels are within expected range
    invalid_labels = [label for label in all_labels if label < 0 or label >= len(DocFigureDataset.CHART_TYPES)]
    if invalid_labels:
        print(f"ERROR: Found invalid labels: {invalid_labels}")
        print(f"Valid range should be [0, {len(DocFigureDataset.CHART_TYPES)-1}]")
        raise ValueError("Invalid labels found in dataset")
    
    # Use the total number of chart types, not just those found in data
    num_classes = len(DocFigureDataset.CHART_TYPES)
    print(f"Creating model with {num_classes} classes")
    
    # Create model
    model = ChartClassifier(num_classes=num_classes).to(device)
    
    # Enhanced loss with label smoothing and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # NEW: Label smoothing for regularization
    print("Using CrossEntropyLoss with label_smoothing=0.1 for better generalization")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # NEW: Enhanced learning rate scheduler - OneCycleLR with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,  # Use provided LR as max
        total_steps=args.epochs * len(train_loader),
        pct_start=0.1,   # 10% warmup
        div_factor=10,   # Initial LR = max_lr / 10
        final_div_factor=1e4,  # Final LR = max_lr / 1e4
    )
    print(f"Using OneCycleLR scheduler with warmup (max_lr={args.lr})")
    
    # Training loop
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'chart_types': DocFigureDataset.CHART_TYPES,
                'num_classes': num_classes
            }, os.path.join(args.work_dir, 'best_model.pth'))
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        
        # Save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'chart_types': DocFigureDataset.CHART_TYPES,
            'num_classes': num_classes
        }, os.path.join(args.work_dir, 'latest_model.pth'))
    
    # Final evaluation
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Classification report - only show classes that appear in validation set
    print("\nClassification Report:")
    unique_labels = sorted(list(set(val_labels)))
    present_chart_types = [DocFigureDataset.CHART_TYPES[i] for i in unique_labels if i < len(DocFigureDataset.CHART_TYPES)]
    
    if len(present_chart_types) > 0:
        print(classification_report(val_labels, val_preds, 
                                  labels=unique_labels,
                                  target_names=present_chart_types))
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.work_dir, 'training_curves.png'))
    plt.show()

if __name__ == '__main__':
    main() 