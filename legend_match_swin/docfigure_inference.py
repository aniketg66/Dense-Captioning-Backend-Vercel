#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DocFigure Chart Classification Inference Script"""

import os
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class ChartClassifier(nn.Module):
    """Chart Type Classifier using ResNet50"""
    
    def __init__(self, num_classes=28):
        super(ChartClassifier, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=True)
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def load_model(model_path, device):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model info
    chart_types = checkpoint.get('chart_types', [])
    num_classes = checkpoint.get('num_classes', len(chart_types))
    
    # Create model
    model = ChartClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model with {num_classes} classes")
    print(f"Model validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    return model, chart_types

def create_transforms():
    """Create inference transforms with improved ImageNet protocol"""
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=Image.BILINEAR),    # Resize to 256 with BILINEAR
        transforms.CenterCrop(224),                              # Central 224×224 crop
        transforms.ToTensor(),                                   # [0,1] pixel range
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    return transform

def predict_single_image(model, image_path, transform, chart_types, device, top_k=3):
    """Predict chart type for a single image"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(chart_types)))
        
        predictions = []
        for i in range(len(top_indices[0])):
            idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            if idx < len(chart_types):
                chart_type = chart_types[idx]
            else:
                chart_type = f"Unknown_Class_{idx}"
            
            predictions.append({
                'chart_type': chart_type,
                'confidence': prob,
                'class_idx': idx
            })
        
        return predictions
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return []

def predict_batch_images(model, image_dir, transform, chart_types, device, output_file=None):
    """Predict chart types for all images in a directory"""
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for file in os.listdir(image_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    results = []
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        predictions = predict_single_image(model, image_path, transform, chart_types, device)
        
        if predictions:
            result = {
                'filename': image_file,
                'predicted_class': predictions[0]['chart_type'],
                'confidence': predictions[0]['confidence'],
                'top_3_predictions': predictions[:3]
            }
            results.append(result)
            
            print(f"{image_file}: {predictions[0]['chart_type']} ({predictions[0]['confidence']:.3f})")
    
    # Save results if output file specified
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='DocFigure Chart Classification Inference')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--image_path', help='Path to single image file')
    parser.add_argument('--image_dir', help='Directory containing images')
    parser.add_argument('--output_file', help='Output JSON file for batch predictions')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    if not args.image_path and not args.image_dir:
        print("Error: Either --image_path or --image_dir must be specified")
        return
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model, chart_types = load_model(args.model_path, device)
    
    print(f"Available chart types ({len(chart_types)}):")
    for i, chart_type in enumerate(chart_types):
        print(f"  {i}: {chart_type}")
    print()
    
    # Create transforms
    transform = create_transforms()
    
    if args.image_path:
        # Single image prediction
        print(f"Predicting chart type for: {args.image_path}")
        predictions = predict_single_image(model, args.image_path, transform, chart_types, device, args.top_k)
        
        if predictions:
            print(f"\nTop {min(args.top_k, len(predictions))} predictions:")
            for i, pred in enumerate(predictions, 1):
                print(f"{i}. {pred['chart_type']}: {pred['confidence']:.4f}")
        else:
            print("Failed to make prediction")
    
    elif args.image_dir:
        # Batch prediction
        print(f"Predicting chart types for images in: {args.image_dir}")
        results = predict_batch_images(model, args.image_dir, transform, chart_types, device, args.output_file)
        
        # Print summary
        print(f"\nProcessed {len(results)} images")
        
        # Count predictions by class
        class_counts = {}
        for result in results:
            pred_class = result['predicted_class']
            class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
        
        print("\nPrediction summary:")
        for chart_type, count in sorted(class_counts.items()):
            print(f"  {chart_type}: {count}")

if __name__ == '__main__':
    main() 