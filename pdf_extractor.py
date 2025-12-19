#!/usr/bin/env python3
"""
PDF Image Extraction for Flask Backend

This script extracts images from PDF files using pdftohtml and returns
the extracted images as base64 data for frontend display.
"""

import os
import json
import subprocess
import shutil
import tempfile
import re
import xml.etree.ElementTree as ET
import base64
from pathlib import Path
from PIL import Image
import io

def parse_pdftohtml_xml(xml_file):
    """Parse the XML file created by pdftohtml to extract text and captions."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        pages_data = {}
        
        for page in root.findall('page'):
            page_num = int(page.get('number', 0))
            page_width = float(page.get('width', 0))
            page_height = float(page.get('height', 0))
            
            texts = []
            for text_elem in page.findall('text'):
                left = float(text_elem.get('left', 0))
                top = float(text_elem.get('top', 0))
                width = float(text_elem.get('width', 0))
                height = float(text_elem.get('height', 0))
                text_content = text_elem.text or ''
                
                texts.append({
                    'bbox': [left, top, width, height],
                    'text': text_content.strip()
                })
            
            pages_data[page_num] = {
                'dimensions': [page_width, page_height],
                'texts': texts
            }
        
        return pages_data
    
    except Exception as e:
        print(f"Error parsing XML file {xml_file}: {e}")
        return {}

def find_captions_in_texts(texts):
    """Find potential figure captions in the text elements."""
    captions = []
    
    for text_data in texts:
        text = text_data['text']
        bbox = text_data['bbox']
        
        # Look for figure captions - more flexible patterns
        patterns = [
            r'^(?:Fig\.?|Figure|FIG\.?)\s*\d+',  # Fig 1, Figure 1, etc.
            r'^(?:Table|TABLE)\s*\d+',          # Table 1, etc.
            r'^\(\s*[a-zA-Z]\s*\)',             # (a), (b), etc.
        ]
        
        for pattern in patterns:
            if re.match(pattern, text, re.IGNORECASE):
                captions.append({
                    'text': text,
                    'bbox': bbox,
                    'type': 'figure' if 'fig' in text.lower() else 'table'
                })
                break
    
    return captions

def get_page_number_from_filename(filename):
    """Extract page number from pdftohtml generated filename."""
    # Examples: page-3_2.jpg -> page 3, page-11_5.jpg -> page 11
    match = re.match(r'page-(\d+)_\d+\.(jpg|png)', filename)
    if match:
        return int(match.group(1))
    return None

def image_to_base64(image_path):
    """Convert image to base64 string for frontend display."""
    try:
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            
            # Validate that we have image data
            if len(img_data) == 0:
                print(f"Error: Empty image file at {image_path}")
                return None
            
            base64_data = base64.b64encode(img_data).decode('utf-8')
            
            # Determine image format
            img = Image.open(io.BytesIO(img_data))
            format_ext = img.format.lower()
            
            # Validate image dimensions
            if img.size[0] < 10 or img.size[1] < 10:
                print(f"Error: Image too small ({img.size[0]}x{img.size[1]}) at {image_path}")
                return None
            
            return f"data:image/{format_ext};base64,{base64_data}"
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def extract_images_from_pdf(pdf_path):
    """Extract images from PDF and return as base64 data."""
    print(f"Processing PDF: {pdf_path}")
    
    temp_dir = Path(tempfile.mkdtemp())
    extracted_images = []
    
    try:
        # Check if pdftohtml is available
        pdftohtml_path = shutil.which('pdftohtml')
        if not pdftohtml_path:
            # Try common installation paths
            common_paths = [
                '/usr/local/bin/pdftohtml',
                '/opt/homebrew/bin/pdftohtml',
                '/usr/bin/pdftohtml'
            ]
            for path in common_paths:
                if os.path.exists(path):
                    pdftohtml_path = path
                    break
        
        if not pdftohtml_path:
            raise Exception("pdftohtml not found. Please install poppler-utils.")
        
        # Step 1: Run pdftohtml to extract figures and text
        print("  Running pdftohtml...")
        cmd = [
            str(pdftohtml_path),
            "-c",  # complex output
            "-hidden",  # include hidden text
            "-xml",  # generate XML with text positions
            str(pdf_path),
            str(temp_dir / "page")
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Error running pdftohtml: {result.stderr}")
            raise Exception(f"pdftohtml failed: {result.stderr}")
        
        # Step 2: Parse XML file for text and captions
        xml_file = temp_dir / "page.xml"
        pages_data = {}
        if xml_file.exists():
            print("  Parsing text and captions...")
            pages_data = parse_pdftohtml_xml(xml_file)
        
        # Step 3: Find all image files created by pdftohtml
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(temp_dir.glob(ext))
        
        # Filter to only include figure images (usually larger JPGs)
        figure_files = []
        for img_file in image_files:
            # Skip small PNG files (usually just text/background)
            if img_file.suffix.lower() == '.png' and img_file.stat().st_size < 10000:
                continue
            # Keep JPG files and larger PNG files
            if img_file.suffix.lower() in ['.jpg', '.jpeg'] or img_file.stat().st_size >= 10000:
                figure_files.append(img_file)
        
        print(f"  Found {len(figure_files)} potential figure images")
        
        # Step 4: Process and convert figures to base64
        extracted_count = 0
        
        for img_file in sorted(figure_files):
            page_num = get_page_number_from_filename(img_file.name)
            if page_num is None:
                continue
            
            extracted_count += 1
            
            # Convert image to base64
            base64_data = image_to_base64(img_file)
            if base64_data and base64_data.startswith('data:image/'):
                # Find caption for this page
                caption_text = ""
                if page_num in pages_data:
                    page_texts = pages_data[page_num]['texts']
                    captions = find_captions_in_texts(page_texts)
                    
                    if captions:
                        # Use the first caption found on this page
                        caption = captions[0]
                        caption_text = caption['text']
                        
                        # Try to get more context (following text)
                        caption_idx = None
                        for i, text_data in enumerate(page_texts):
                            if text_data['text'] == caption_text:
                                caption_idx = i
                                break
                        
                        if caption_idx is not None:
                            # Collect text following the caption
                            full_caption = [caption_text]
                            for i in range(caption_idx + 1, min(caption_idx + 5, len(page_texts))):
                                next_text = page_texts[i]['text'].strip()
                                if next_text and not re.match(r'^(?:Fig\.?|Figure|Table)', next_text, re.IGNORECASE):
                                    full_caption.append(next_text)
                                else:
                                    break
                            caption_text = ' '.join(full_caption)
                
                extracted_images.append({
                    'url': base64_data,
                    'filename': f"figure_{extracted_count:03d}.jpg",
                    'page': page_num,
                    'caption': caption_text if caption_text else f"Figure {extracted_count} (no caption found)"
                })
                
                print(f"    Extracted figure_{extracted_count:03d}.jpg from page {page_num}")
        
        print(f"  Successfully extracted {len(extracted_images)} figures")
        return extracted_images
        
    except Exception as e:
        print(f"  Error processing PDF: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    finally:
        # Clean up temporary files
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Test the extraction function
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pdf_extractor.py <pdf_file>")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    if not os.path.exists(pdf_file):
        print(f"PDF file not found: {pdf_file}")
        sys.exit(1)
    
    try:
        images = extract_images_from_pdf(pdf_file)
        print(f"Extracted {len(images)} images")
        for i, img in enumerate(images):
            print(f"  {i+1}. {img['filename']} (page {img['page']})")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 