#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explore DocFigure Dataset Structure"""

import os
import csv
from collections import Counter
import argparse

def explore_annotations(ann_file):
    """Explore annotation file structure"""
    print(f"Exploring annotation file: {ann_file}")
    
    if not os.path.exists(ann_file):
        print(f"Error: File {ann_file} does not exist!")
        return
    
    categories = []
    filenames = []
    
    with open(ann_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for line_num, row in enumerate(reader, 1):
            if len(row) < 2:
                print(f"Warning: Line {line_num} has insufficient columns: {row}")
                continue
                
            filename = row[0].strip()
            category = row[1].strip()
            
            filenames.append(filename)
            categories.append(category)
    
    print(f"Total annotations: {len(categories)}")
    print(f"Unique categories: {len(set(categories))}")
    print(f"Unique filenames: {len(set(filenames))}")
    
    # Category distribution
    category_counts = Counter(categories)
    print(f"\nCategory distribution (top 20):")
    for category, count in category_counts.most_common(20):
        print(f"  {category}: {count}")
    
    # Check for duplicates
    filename_counts = Counter(filenames)
    duplicates = [f for f, count in filename_counts.items() if count > 1]
    if duplicates:
        print(f"\nFound {len(duplicates)} duplicate filenames:")
        for dup in duplicates[:10]:  # Show first 10
            print(f"  {dup}: {filename_counts[dup]} times")
    
    return categories, filenames

def check_images(images_dir, filenames):
    """Check if image files exist"""
    print(f"\nChecking images in: {images_dir}")
    
    if not os.path.exists(images_dir):
        print(f"Error: Directory {images_dir} does not exist!")
        return
    
    # Get all image files in directory
    actual_files = set(os.listdir(images_dir))
    expected_files = set(filenames)
    
    print(f"Files in annotation: {len(expected_files)}")
    print(f"Files in directory: {len(actual_files)}")
    
    # Find missing files
    missing_files = expected_files - actual_files
    if missing_files:
        print(f"\nMissing {len(missing_files)} files:")
        for file in list(missing_files)[:10]:  # Show first 10
            print(f"  {file}")
    
    # Find extra files
    extra_files = actual_files - expected_files
    if extra_files:
        print(f"\nExtra {len(extra_files)} files not in annotations:")
        for file in list(extra_files)[:10]:  # Show first 10
            print(f"  {file}")
    
    # File type analysis
    extensions = Counter()
    for file in actual_files:
        ext = os.path.splitext(file)[1].lower()
        extensions[ext] += 1
    
    print(f"\nFile extensions:")
    for ext, count in extensions.most_common():
        print(f"  {ext}: {count}")

def compare_docfigure_categories():
    """Compare with official DocFigure 28 categories"""
    
    official_categories = [
        'Line graph', 'Natural image', 'Table', '3D objects', 'Bar plot', 
        'Scatter plot', 'Medical image', 'Sketch', 'Geographic map', 'Flow chart',
        'Heat map', 'Mask', 'Block diagram', 'Venn diagram', 'Confusion matrix',
        'Histogram', 'Box plot', 'Vector plot', 'Pie chart', 'Surface plot',
        'Algorithm', 'Contour plot', 'Tree diagram', 'Bubble chart', 'Polar plot',
        'Area chart', 'Pareto chart', 'Radar chart'
    ]
    
    print(f"\nOfficial DocFigure categories ({len(official_categories)}):")
    for i, cat in enumerate(official_categories, 1):
        print(f"  {i:2d}. {cat}")
    
    return official_categories

def main():
    parser = argparse.ArgumentParser(description='Explore DocFigure Dataset')
    parser.add_argument('--images_dir', 
                       default='/content/drive/MyDrive/Research Summer 2025/Dense Captioning Toolkit/CHART-Classification/images',
                       help='Directory containing images')
    parser.add_argument('--train_ann', 
                       default='/content/drive/MyDrive/Research Summer 2025/Dense Captioning Toolkit/CHART-Classification/annotation',
                       help='Training annotations file')
    parser.add_argument('--val_ann', 
                       default='/content/drive/MyDrive/Research Summer 2025/Dense Captioning Toolkit/CHART-Classification/val.txt',
                       help='Validation annotations file')
    
    args = parser.parse_args()
    
    print("="*80)
    print("DOCFIGURE DATASET EXPLORATION")
    print("="*80)
    
    # Show official categories
    official_categories = compare_docfigure_categories()
    
    # Explore training annotations
    print("\n" + "="*60)
    print("TRAINING ANNOTATIONS")
    print("="*60)
    train_categories, train_filenames = explore_annotations(args.train_ann)
    
    # Explore validation annotations
    print("\n" + "="*60)
    print("VALIDATION ANNOTATIONS")
    print("="*60)
    val_categories, val_filenames = explore_annotations(args.val_ann)
    
    # Check training images
    if train_filenames:
        print("\n" + "="*60)
        print("TRAINING IMAGES CHECK")
        print("="*60)
        check_images(args.images_dir, train_filenames)
    
    # Check validation images
    if val_filenames:
        print("\n" + "="*60)
        print("VALIDATION IMAGES CHECK")
        print("="*60)
        check_images(args.images_dir, val_filenames)
    
    # Compare categories with official list
    if train_categories or val_categories:
        print("\n" + "="*60)
        print("CATEGORY COMPARISON")
        print("="*60)
        
        all_found_categories = set(train_categories + val_categories)
        official_set = set(official_categories)
        
        print(f"Found categories: {len(all_found_categories)}")
        print(f"Official categories: {len(official_set)}")
        
        # Categories in data but not in official list
        extra_cats = all_found_categories - official_set
        if extra_cats:
            print(f"\nExtra categories not in official list ({len(extra_cats)}):")
            for cat in sorted(extra_cats):
                print(f"  {cat}")
        
        # Official categories not found in data
        missing_cats = official_set - all_found_categories
        if missing_cats:
            print(f"\nOfficial categories not found in data ({len(missing_cats)}):")
            for cat in sorted(missing_cats):
                print(f"  {cat}")
        
        # Matching categories
        matching_cats = all_found_categories & official_set
        print(f"\nMatching categories ({len(matching_cats)}):")
        for cat in sorted(matching_cats):
            print(f"  {cat}")

if __name__ == '__main__':
    main() 