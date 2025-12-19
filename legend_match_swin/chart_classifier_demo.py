#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chart Type Classification Demo Script - Process Multiple Images"""

import os
import glob
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Enable inline plotting for Colab/Jupyter
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    from IPython.display import Image as IPImage, display
    COLAB_AVAILABLE = True
except (NameError, AttributeError):
    # Not in Jupyter/Colab environment
    COLAB_AVAILABLE = False

class ChartClassifier(nn.Module):
    """Chart Type Classifier using ResNet50"""
    
    def __init__(self, num_classes=5):
        super(ChartClassifier, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=True)
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def load_model(checkpoint_path, device):
    """Load the trained model from checkpoint"""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get chart types and number of classes
    chart_types = checkpoint.get('chart_types', ['dot', 'horizontal_bar', 'line', 'scatter', 'vertical_bar'])
    num_classes = len(chart_types)
    
    # Create model with correct number of classes
    model = ChartClassifier(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, chart_types

def preprocess_image(image_path):
    """Preprocess image for inference"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and convert image
    image = Image.open(image_path).convert('RGB')
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor, image

def predict(model, image_tensor, chart_types, device, unknown_threshold=0.5):
    """Run inference on preprocessed image"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Get prediction
        pred_prob, pred_idx = torch.max(probabilities, 1)
        pred_label = pred_idx.item()
        pred_score = pred_prob.item()
        pred_class = chart_types[pred_label]
        
        # Handle unknown chart types (low confidence predictions)
        # If confidence is below threshold, classify as "pie"
        if pred_score < unknown_threshold:
            pred_class = "pie"
            print(f"Low confidence ({pred_score:.3f} < {unknown_threshold}), classifying as 'pie' chart")
        
        # Get all scores
        all_scores = probabilities.squeeze().cpu().numpy()
    
    return {
        'pred_label': pred_label,
        'pred_score': pred_score,
        'pred_class': pred_class,
        'pred_scores': all_scores,
        'original_pred_class': chart_types[pred_label]  # Keep original prediction
    }

def demo_chart_classification(checkpoint_path, images_dir, num_images=6, unknown_threshold=0.5):
    """Demo function to classify multiple chart images"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model, chart_types = load_model(checkpoint_path, device)
    print(f"Model loaded successfully. Chart types: {chart_types}")
    
    # Find image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        image_files.extend(glob.glob(os.path.join(images_dir, ext.upper())))
    
    image_files.sort()  # Sort for consistent ordering
    
    if len(image_files) == 0:
        print(f"No image files found in {images_dir}")
        return
    
    # Limit to requested number of images
    image_files = image_files[:num_images]
    print(f"Found {len(image_files)} images to process")
    
    # Process images and collect results
    results = []
    
    for i, image_path in enumerate(image_files):
        print(f"\nProcessing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        try:
            # Preprocess image
            image_tensor, original_image = preprocess_image(image_path)
            
            # Run inference
            result = predict(model, image_tensor, chart_types, device, unknown_threshold)
            
            # Add image info to result
            result['image_path'] = image_path
            result['image_name'] = os.path.basename(image_path)
            result['original_image'] = original_image
            
            results.append(result)
            
            print(f"  Original prediction: {result['original_pred_class']} (confidence: {result['pred_score']:.3f})")
            print(f"  Final prediction: {result['pred_class']}")
            
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
    
    # Create visualization
    if results:
        display_individual_images(results, chart_types)
        create_visualization(results, chart_types)
    
    return results

def display_individual_images(results, chart_types):
    """Display each image individually with its prediction"""
    
    print("\n" + "="*60)
    print("DISPLAYING IMAGES")
    print("="*60)
    
    for i, result in enumerate(results):
        print(f"\nImage {i+1}/{len(results)}: {result['image_name']} → {result['pred_class']} ({result['pred_score']:.3f})")
        
        # Use matplotlib for consistent display
        plt.figure(figsize=(8, 6))
        plt.imshow(result['original_image'])
        plt.title(f"{result['image_name']}\nPredicted: {result['pred_class']} (confidence: {result['pred_score']:.3f})", 
                 fontsize=14, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()  # Force the notebook to render the figure

def create_visualization(results, chart_types):
    """Create a visualization grid showing images and their predictions"""
    
    num_images = len(results)
    cols = min(3, num_images)  # Max 3 columns
    rows = (num_images + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 6*rows))
    
    # Handle single row case
    if rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    for i, result in enumerate(results):
        ax = axes[i]
        
        # Display image
        ax.imshow(result['original_image'])
        
        # Create title with prediction info
        title = f"{result['image_name']}\n"
        title += f"Predicted: {result['pred_class']}\n"
        title += f"Confidence: {result['pred_score']:.3f}"
        
        if result['pred_class'] != result['original_pred_class']:
            title += f"\n(Original: {result['original_pred_class']})"
        
        ax.set_title(title, fontsize=10, pad=10)
        ax.axis('off')
        
        # Add confidence bar
        scores = result['pred_scores']
        max_idx = result['pred_label']
        
        # Create mini bar chart in corner
        bar_height = 0.02
        bar_width = 0.3
        bar_x = 0.65
        bar_y = 0.02
        
        for j, (chart_type, score) in enumerate(zip(chart_types, scores)):
            color = 'red' if j == max_idx else 'lightblue'
            width = bar_width * score
            y_pos = bar_y + j * (bar_height + 0.005)
            
            # Add rectangle for score bar
            rect = plt.Rectangle((bar_x, y_pos), width, bar_height, 
                               transform=ax.transAxes, color=color, alpha=0.7)
            ax.add_patch(rect)
            
            # Add text label
            ax.text(bar_x - 0.01, y_pos + bar_height/2, f"{chart_type[:4]}", 
                   transform=ax.transAxes, fontsize=8, va='center', ha='right')
    
    # Hide unused subplots
    for i in range(len(results), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Chart Type Classification Demo', fontsize=16, y=0.98)
    plt.show()  # Render the entire grid
    
    # Print summary
    print("\n" + "="*60)
    print("CLASSIFICATION SUMMARY")
    print("="*60)
    
    for i, result in enumerate(results):
        print(f"{i+1:2d}. {result['image_name']:<30} → {result['pred_class']:<15} ({result['pred_score']:.3f})")
    
    # Print class distribution
    pred_counts = {}
    for result in results:
        pred_class = result['pred_class']
        pred_counts[pred_class] = pred_counts.get(pred_class, 0) + 1
    
    print("\nPredicted class distribution:")
    for chart_type, count in sorted(pred_counts.items()):
        print(f"  {chart_type}: {count}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Chart Type Classification Demo')
    parser.add_argument('--checkpoint', 
                       default='work_dirs/chart_classifier/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--images_dir', 
                       default='legend_data/train/images',
                       help='Directory containing images')
    parser.add_argument('--num_images', type=int, default=6,
                       help='Number of images to process')
    parser.add_argument('--unknown_threshold', type=float, default=0.5,
                       help='Confidence threshold for unknown classification')
    
    args = parser.parse_args()
    
    # Check if paths exist
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please train the model first or provide correct checkpoint path")
        return
    
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory not found at {args.images_dir}")
        return
    
    # Run demo
    results = demo_chart_classification(
        checkpoint_path=args.checkpoint,
        images_dir=args.images_dir,
        num_images=args.num_images,
        unknown_threshold=args.unknown_threshold
    )
    
    print(f"\nDemo completed! Processed {len(results)} images.")

if __name__ == '__main__':
    main() 