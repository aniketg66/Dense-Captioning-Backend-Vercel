import sys
import os
import re

# Add mmcv to Python path (only if it exists, for local development)
mmcv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mmcv'))
if os.path.exists(mmcv_path) and mmcv_path not in sys.path:
    sys.path.insert(0, mmcv_path)

from flask import Flask, render_template, request, jsonify, make_response
from flask_cors import CORS
import json
from datetime import datetime
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import uuid
import io
import colorsys
import traceback
from openai import OpenAI
from dotenv import load_dotenv
import tempfile
from werkzeug.utils import secure_filename
import requests

# Heavy ML imports - make optional for Vercel deployment
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: whisper not available")

try:
    import torch
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch/torchvision not available")

try:
    from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available")

try:
    from skimage import measure
    from skimage.measure import label, find_contours, approximate_polygon
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available")

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: easyocr not available")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python not available")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not available")

# Import PDF extraction functionality
from pdf_extractor import extract_images_from_pdf
from segregate_image import segment_image

# Load environment variables from backend directory .env
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client (optional - only if API key is provided)
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    client = None
    print("Warning: OPENAI_API_KEY not set. GPT refinement will be disabled.")

# State file for processed images per user
# Use /tmp in Vercel environment
if os.environ.get("VERCEL"):
    PROCESSED_IMAGES_FILE = "/tmp/processed_images.json"
else:
    PROCESSED_IMAGES_FILE = "processed_images.json"

# Initialize Whisper model (lazy loading for Vercel)
whisper_model = None
def get_whisper_model():
    global whisper_model
    if not WHISPER_AVAILABLE:
        return None
    if whisper_model is None:
        try:
            whisper_model = whisper.load_model("base")
        except Exception as e:
            print(f"Warning: Could not load Whisper model: {e}")
            whisper_model = False  # Mark as failed
    return whisper_model if whisper_model else None

# Initialize MaskFormer (lazy loading for Vercel)
maskformer_processor = None
maskformer_model = None
device = None

def get_maskformer():
    global maskformer_processor, maskformer_model, device
    if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
        return None
    if maskformer_model is None:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            maskformer_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-ade")
            maskformer_model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade")
            maskformer_model.to(device)
            maskformer_model.eval()
            print("MaskFormer model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load MaskFormer model: {e}")
            maskformer_model = False  # Mark as failed
    return maskformer_model if maskformer_model else None

# Define image transform for Mask R-CNN
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

# ADE20k class labels
ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass",
    "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair",
    "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair",
    "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion", "base",
    "box", "column", "signboard", "chest", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pool", "pillow", "screen",
    "stairway", "river", "bridge", "bookcase", "blind", "coffee table", "toilet", "flower", "book",
    "hill", "bench", "countertop", "stove", "palm", "kitchen island", "computer", "horse", "unknown"
]

print("MaskFormer model loaded successfully")

# =============================================================================
# HUGGINGFACE SPACE INTEGRATION (No local ML models for segmentation)
# =============================================================================
# Using HuggingFace Space for MedSAM segmentation instead of local models
# This allows deployment on platforms without GPU/large model requirements

from utils.huggingface_client import HuggingFaceSegmentationClient, get_huggingface_client

# HuggingFace Space URL for MedSAM
HUGGINGFACE_SPACE_URL = "Aniketg6/medsam-inference"

# Initialize HuggingFace client (lazy initialization - connects on first use)
hf_segmentation_client = None

def get_segmentation_client():
    """Get or create the HuggingFace segmentation client"""
    global hf_segmentation_client
    if hf_segmentation_client is None:
        hf_segmentation_client = HuggingFaceSegmentationClient(HUGGINGFACE_SPACE_URL)
    return hf_segmentation_client

print(f"Segmentation will use HuggingFace Space: {HUGGINGFACE_SPACE_URL}")
print("No local SAM/MedSAM models will be loaded")

# Create necessary directories
# Use /tmp for Vercel serverless environment, otherwise use local directories
# Check for Vercel environment (can be set by Vercel or manually for testing)
VERCEL_ENV = os.environ.get("VERCEL") or os.environ.get("VERCEL_ENV")
if VERCEL_ENV:
    BASE_DIR = "/tmp"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
TRANSCRIPTIONS_DIR = os.path.join(BASE_DIR, "transcriptions")
STATIC_IMAGES_DIR = os.path.join(BASE_DIR, "static", "images")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)
os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)

# Sample images
SAMPLE_IMAGES = [
    "object1.jpg",     # First object image
    "object2.jpg",     # Second object image
    "landscape1.jpg",  # Landscape images
    "landscape2.jpg",
    "landscape3.jpg",
    "text.jpg"
]

# Initialize EasyOCR reader
ocr_reader = easyocr.Reader(['en'])  # Initialize for English

# HF Space configuration using gradio_client - Direct URL approach
HF_SPACE_URL = "https://hanszhu-dense-captioning-platform.hf.space"

# Enhanced 21-class categories for comprehensive chart element detection
ENHANCED_CLASS_NAMES = [
    'title', 'subtitle', 'x-axis', 'y-axis', 'x-axis-label', 'y-axis-label',
    'x-tick-label', 'y-tick-label', 'legend', 'legend-title', 'legend-item',
    'data-point', 'data-line', 'data-bar', 'data-area', 'grid-line',
    'axis-title', 'tick-label', 'data-label', 'legend-text', 'plot-area'
]

def perform_local_ocr(img_np):
    """
    Perform local OCR using EasyOCR for text detection
    Returns list of text elements with bbox and confidence
    """
    try:
        # Use the existing EasyOCR reader from the global scope
        results = ocr_reader.readtext(img_np)
        
        text_elements = []
        for (bbox, text, confidence) in results:
            # Convert bbox format from EasyOCR to our format
            # EasyOCR bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            # Our format is [x1, y1, x2, y2]
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            bbox_formatted = [
                min(x_coords),  # x1
                min(y_coords),  # y1
                max(x_coords),  # x2
                max(y_coords)   # y2
            ]
            
            text_elements.append({
                'text': text,
                'bbox': bbox_formatted,
                'confidence': confidence,
                'element_type': 'text',
                'label': 'text'
            })
        
        print(f"Local OCR detected {len(text_elements)} text elements")
        return text_elements
        
    except Exception as e:
        print(f"Error in local OCR: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return []

def call_hf_space_api(image_path):
    """
    Call the Dense Captioning Platform API using gradio_client to analyze a scientific image
    Returns the raw API response
    """
    try:
        from gradio_client import Client, handle_file
        
        print(f"Calling Dense Captioning Platform API: {HF_SPACE_URL}")
        
        # Initialize client with direct URL (working approach)
        client = Client("https://hanszhu-dense-captioning-platform.hf.space")
        
        # Call the predict function using the working approach
        try:
            result = client.predict(
                image=handle_file(image_path),
                fn_index=0
            )
        except Exception as e:
            print(f"API call failed: {e}")
            raise e
        
        print(f"API result: {result}")
        return result
            
    except Exception as e:
        print(f"Error calling Dense Captioning Platform API: {str(e)}")
        return None

def parse_hf_space_response(hf_response):
    """
    Parse the HF Space API response from gradio_client and convert it to the format expected by the frontend
    """
    if not hf_response:
        return None
    
    try:
        # The gradio_client response might be a string or dict
        if isinstance(hf_response, str):
            # Try to parse as JSON if it's a string
            try:
                import json
                hf_response = json.loads(hf_response)
            except json.JSONDecodeError:
                # If it's not JSON, it might be the raw DetDataSample string
                pass
        
        # If it's still a string, it's likely the DetDataSample format
        if isinstance(hf_response, str):
            return parse_det_data_sample_string(hf_response)
        
        # If it's a dict, extract the data
        if isinstance(hf_response, dict):
            # Extract chart type information
            chart_type_id = hf_response.get('chart_type_id', None)
            chart_type_label = hf_response.get('chart_type_label', None)
            chart_type_name = chart_type_label if chart_type_label else f"chart_type_{chart_type_id}" if chart_type_id else None
            
            # Parse element results (DetDataSample format) - these are chart elements (axes, titles, legends, etc.)
            element_result = hf_response.get('element_result', '')
            chart_elements = parse_det_data_sample_string(element_result, element_type_prefix="chart_element") if element_result else []
            
            # Parse datapoint results (DetDataSample format) - these are actual data points (bars, points, lines, etc.)
            datapoint_result = hf_response.get('datapoint_result', '')
            data_points = parse_det_data_sample_string(datapoint_result, element_type_prefix="data_point") if datapoint_result else []
            
            # Apply thresholds to filter elements
            element_threshold = 0.4  # General chart elements
            datapoint_threshold = 0.4  # Data points (not bars)
            databar_threshold = 0.01   # Data bars (very low threshold)
            
            # Filter chart elements by confidence
            filtered_chart_elements = []
            for elem in chart_elements:
                element_type = elem.get('element_type', '').lower()
                confidence = elem.get('confidence', 0)
                
                # Use different thresholds based on element type
                if 'bar' in element_type:
                    threshold = databar_threshold  # Very low for bars
                elif 'data' in element_type and 'point' in element_type:
                    threshold = datapoint_threshold  # Higher for data points
                else:
                    threshold = element_threshold  # General elements
                
                if confidence >= threshold:
                    filtered_chart_elements.append(elem)
            
            # Filter data points by confidence
            filtered_data_points = [dp for dp in data_points if dp.get('confidence', 0) >= datapoint_threshold]
            
            # Combine all results (text elements will be added by local OCR)
            analysis_results = {
                'chart_type': chart_type_name,
                'chart_type_confidence': 0.9 if chart_type_name else None,
                'chart_elements': filtered_chart_elements,
                'data_points': filtered_data_points,
                'text_elements': [],  # Will be filled by local OCR
                'status': hf_response.get('status', 'unknown'),
                'processing_time': hf_response.get('processing_time', 0)
            }
            
            print(f"[DEBUG] Parsed analysis results: chart_type={chart_type_name}, elements={len(analysis_results['chart_elements'])}, datapoints={len(analysis_results['data_points'])}")
            
            return analysis_results
        
    except Exception as e:
        print(f"Error parsing HF Space response: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return None

def parse_det_data_sample_string(det_data_sample_str, element_type_prefix="element"):
    """
    Parse a DetDataSample string format and convert it to the format expected by the frontend
    
    Args:
        det_data_sample_str: The DetDataSample string to parse
        element_type_prefix: Prefix for element types ("chart_element" or "data_point")
    """
    try:
        if not det_data_sample_str or not isinstance(det_data_sample_str, str):
            return []
        
        # Extract bounding boxes, labels, and scores from the string
        bboxes = []
        labels = []
        scores = []
        
        # Parse bboxes - look for tensor format with multiple bbox arrays
        # The bboxes are in format: tensor([[x1, y1, x2, y2], [x1, y1, x2, y2], ...])
        bbox_pattern = r'bboxes:\s*tensor\(\[(.*?)\]\)'
        bbox_matches = re.findall(bbox_pattern, det_data_sample_str, re.DOTALL)
        
        if bbox_matches:
            bbox_content = bbox_matches[0]
            # Split by individual bbox arrays
            bbox_array_pattern = r'\[([\d\.,\s\-]+)\]'
            bbox_arrays = re.findall(bbox_array_pattern, bbox_content)
            
            for bbox_str in bbox_arrays:
                bbox_str = bbox_str.strip()
                if bbox_str:
                    try:
                        # Split by comma and convert to floats
                        bbox_values = [float(x.strip()) for x in bbox_str.split(',') if x.strip()]
                        if len(bbox_values) == 4:
                            bboxes.append(bbox_values)
                    except ValueError:
                        continue
        
        # Parse labels - look for tensor format
        label_pattern = r'labels:\s*tensor\(\[([\d\s,]+)\]\)'
        label_matches = re.findall(label_pattern, det_data_sample_str)
        if label_matches:
            label_str = label_matches[0]
            labels = [int(x.strip()) for x in label_str.split(',') if x.strip()]
        
        # Parse scores - look for tensor format
        score_pattern = r'scores:\s*tensor\(\[([\d\.\s,]+)\]\)'
        score_matches = re.findall(score_pattern, det_data_sample_str)
        if score_matches:
            score_str = score_matches[0]
            scores = [float(x.strip()) for x in score_str.split(',') if x.strip()]
        
        # Create elements with proper type prefix
        elements = []
        for i in range(min(len(bboxes), len(labels), len(scores))):
            label_idx = labels[i]
            if element_type_prefix == "chart_element":
                # Use enhanced class names for chart elements
                element_type = ENHANCED_CLASS_NAMES[label_idx] if label_idx < len(ENHANCED_CLASS_NAMES) else f'chart_element_{label_idx}'
            else:
                # Use data point prefix for data points
                element_type = f'{element_type_prefix}_{label_idx}'
            elements.append({
                'element_type': element_type,
                'bbox': bboxes[i],
                'confidence': scores[i],
                'label': element_type
            })
        
        print(f"[DEBUG] Parsed {len(elements)} {element_type_prefix} elements from DetDataSample string")
        return elements
        
    except Exception as e:
        print(f"Error parsing DetDataSample string: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return []

# Initialize Scientific Chart Analyzer (keeping for compatibility but not using for processing)
try:
    from science_analyzer import ScientificImageAnalyzer
    chart_analyzer = ScientificImageAnalyzer()
    print("Scientific Chart Analyzer initialized successfully (for compatibility)")
except Exception as e:
    print(f"Warning: Could not initialize Scientific Chart Analyzer: {e}")
    chart_analyzer = None

def load_processed_images():
    if os.path.exists(PROCESSED_IMAGES_FILE):
        with open(PROCESSED_IMAGES_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_processed_images(data):
    with open(PROCESSED_IMAGES_FILE, 'w') as f:
        json.dump(data, f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/get-image')
def get_image():
    user_id = request.cookies.get('user_id')
    if not user_id:
        user_id = str(uuid.uuid4())
        response = make_response(jsonify({"image": SAMPLE_IMAGES[0]}))
        response.set_cookie('user_id', user_id)

        processed_data = load_processed_images()
        processed_data[user_id] = []
        save_processed_images(processed_data)
    else:
        processed_data = load_processed_images()
        processed_images = processed_data.get(user_id, [])

        next_image = None
        for image in SAMPLE_IMAGES:
            if image not in processed_images:
                next_image = image
                break

        if next_image is None:
            processed_data[user_id] = []
            save_processed_images(processed_data)
            next_image = SAMPLE_IMAGES[0]
            
        response = make_response(jsonify({"image": next_image}))

    return response

@app.route('/api/pre-segment', methods=['POST'])
def pre_segment():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Get image dimensions
        height, width = image_array.shape[:2]
        print(f"Processing image of size: {width}x{height}")
        
        # Preprocess image for MaskFormer
        maskformer_model = get_maskformer()
        if not maskformer_model or not maskformer_processor:
            return jsonify({'error': 'MaskFormer model not available'}), 503
        
        inputs = maskformer_processor(images=image, return_tensors="pt").to(device)
        
        # Run MaskFormer prediction
        with torch.no_grad():
            outputs = maskformer_model(**inputs)
        
        # Get the segmentation results
        predicted_semantic_map = maskformer_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(height, width)]
        )[0]
        
        # Move the tensor to CPU and convert to NumPy
        predicted_semantic_map = predicted_semantic_map.cpu().numpy()
        
        # Get unique regions
        unique_regions = []
        used_pixels = np.zeros((height, width), dtype=bool)
        
        # Process each unique class in the semantic map
        unique_classes = np.unique(predicted_semantic_map)
        for class_id in unique_classes:
            if class_id == 0:  # Skip background class
                continue
                
            # Create mask for this class
            mask = predicted_semantic_map == class_id
            
            # Skip if mask is too small
            if np.sum(mask) < 100:  # Minimum 100 pixels
                continue
            
            # Get class name
            class_name = ADE20K_CLASSES[class_id] if class_id < len(ADE20K_CLASSES) else "unknown"
            
            # Get the raw mask as a binary array
            binary_mask = mask.astype(np.uint8)
            
            # Generate a unique color for this region
            color = [int(c * 255) for c in colorsys.hsv_to_rgb(class_id / len(ADE20K_CLASSES), 0.8, 0.8)]
            
            # Add region to list with mask and color
            unique_regions.append({
                'id': len(unique_regions),
                'class': class_name,
                'score': 1.0,  # Semantic segmentation doesn't provide confidence scores
                'mask': binary_mask.tolist(),
                'color': color
            })
            
            # Mark pixels as used
            used_pixels = np.logical_or(used_pixels, mask)
        
        # Create background regions for unused pixels
        labeled_image = label(~used_pixels)
        for region_id in range(1, labeled_image.max() + 1):
            mask = labeled_image == region_id
            
            # Skip if region is too small
            if np.sum(mask) < 100:  # Minimum 100 pixels
                continue
            
            # Get the raw mask as a binary array
            binary_mask = mask.astype(np.uint8)
            
            # Generate a unique color for this background region
            color = [int(c * 255) for c in colorsys.hsv_to_rgb(0.5, 0.3, 0.8)]  # Grayish color
            
            # Add background region to list with mask and color
            unique_regions.append({
                'id': len(unique_regions),
                'class': 'background',
                'score': 1.0,
                'mask': binary_mask.tolist(),
                'color': color
            })
        
        print(f"Generated {len(unique_regions)} unique regions")
        return jsonify({'regions': unique_regions})
        
    except Exception as e:
        print(f"Error in pre-segment: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/segment', methods=['POST'])
def segment():
    """Segment using point via HuggingFace Space API"""
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Get point coordinates
        x = int(data['x'])
        y = int(data['y'])
        
        # Check if point is in a background region
        is_background = True
        for region in data.get('regions', []):
            if region['class'] != 'background':
                # Check if point is inside any non-background region
                for polygon in region['polygons']:
                    if point_in_polygon(x, y, polygon):
                        is_background = False
                        break
                if not is_background:
                    break
        
        if not is_background:
            return jsonify({'error': 'Point is not in a background region'}), 400
        
        # Get HuggingFace segmentation client and load the image
        segmentation_client = get_segmentation_client()
        segmentation_client.load_image_from_array(image_array)
        
        # Call HuggingFace API for point-based segmentation
        result = segmentation_client.segment_with_points([[x, y]], [1])
        
        if not result or not result.get('masks'):
            return jsonify({'error': 'Segmentation failed'}), 500
        
        # Get the best mask (first one, as HuggingFace returns the best already)
        best_mask_data = result['masks'][0]['mask']
        best_mask = np.array(best_mask_data, dtype=np.uint8)
        best_score = result['confidences'][0] if result.get('confidences') else 0.9
        
        # Get contours for visualization
        contours = find_contours(best_mask, 0.5)
        polygons = []
        for contour in contours:
            # Simplify polygon
            contour = approximate_polygon(contour, tolerance=1.0)
            # Ensure polygon is closed
            if len(contour) > 0 and not np.array_equal(contour[0], contour[-1]):
                contour = np.vstack((contour, contour[0]))
            # Fix diagonal reflection by swapping x and y coordinates
            contour = np.flip(contour, axis=1)
            polygons.append(contour.tolist())
        
        # Generate a unique color for this region
        color = [int(c * 255) for c in colorsys.hsv_to_rgb(0.5, 0.3, 0.8)]  # Grayish color
        
        return jsonify({
            'id': -1,  # New region
            'class': 'background',
            'score': float(best_score),
            'polygons': polygons,
            'color': color
        })
        
    except Exception as e:
        print(f"Error in segment: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def point_in_polygon(x, y, polygon):
    """Check if a point is inside a polygon using ray casting algorithm."""
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

@app.route('/api/detect', methods=['POST'])
def detect_objects():
    print("Received detection request")
    data = request.json
    image_data = data.get('image')
    
    try:
        # Convert base64 image to numpy array
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        image_height, image_width = np.array(image).shape[:2]
        
        # Preprocess image for MaskFormer
        maskformer_model = get_maskformer()
        if not maskformer_model or not maskformer_processor:
            return jsonify({"error": "MaskFormer model not available"}), 503
        
        inputs = maskformer_processor(images=image, return_tensors="pt").to(device)
        
        # Run MaskFormer prediction
        with torch.no_grad():
            outputs = maskformer_model(**inputs)
        
        # Get class predictions and scores
        class_logits = outputs.class_queries_logits
        mask_logits = outputs.masks_queries_logits
        
        # Process results
        detections = []
        for idx in range(len(class_logits[0])):
            # Get class prediction and score
            class_scores = torch.softmax(class_logits[0][idx], dim=0)
            class_id = torch.argmax(class_scores).item()
            score = float(class_scores[class_id])
            
            # Skip if score is too low
            if score < 0.5:  # Confidence threshold
                continue
            
            # Get mask prediction
            mask = torch.sigmoid(mask_logits[0][idx]).cpu().numpy()
            
            # Skip if mask is too small
            if np.sum(mask > 0.5) < 100:  # Minimum 100 pixels
                continue
            
            # Get class name
            class_name = ADE20K_CLASSES[class_id] if class_id < len(ADE20K_CLASSES) else "unknown"
            
            # Get bounding box from mask
            y_indices, x_indices = np.where(mask > 0.5)
            if len(y_indices) > 0 and len(x_indices) > 0:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                
                detections.append({
                    "box": [float(x_min), float(y_min), float(x_max), float(y_max)],
                    "confidence": score,
                    "class": class_name
                })
        
        return jsonify({
            "detections": detections
        })

    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/ocr', methods=['POST'])
def perform_ocr():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to cv2 format
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Perform OCR
        results = ocr_reader.readtext(image)
        
        # Format results
        ocr_data = []
        for (bbox, text, prob) in results:
            if prob > 0.5:  # Filter low confidence results
                # Convert bbox points to relative coordinates
                height, width = image.shape[:2]
                bbox_relative = [
                    [point[0]/width, point[1]/height] for point in bbox
                ]
                
                ocr_data.append({
                    'text': text,
                    'confidence': float(prob),
                    'bbox': bbox_relative
                })
        
        return jsonify({
            'success': True,
            'ocr_data': ocr_data
        })
        
    except Exception as e:
        print(f"Error in OCR processing: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def refine_transcription_with_gpt(transcription, image_filename, ocr_text=None):
    if client is None:
        print("OpenAI client not initialized. Skipping GPT refinement.")
        return None
    try:
        # Include OCR text in the prompt if available
        ocr_context = ""
        if ocr_text:
            ocr_context = f"\nThe image contains the following text detected by OCR:\n{ocr_text}"
        
        prompt = f"""As an AI trained to generate detailed image descriptions, please refine and expand the following transcription to make it more suitable for training visual language models. The transcription is about the image named '{image_filename}'.{ocr_context}

Original transcription:
{transcription}

Please provide a refined version that:
1. Is more detailed and descriptive
2. Uses precise and specific language
3. Captures spatial relationships between objects
4. Includes relevant attributes (colors, sizes, textures)
5. Maintains a natural, flowing narrative
6. Incorporates any relevant text found in the image (from OCR) if it adds context
7. Focuses on visual elements that would be valuable for training vision-language models

Refined description:"""

        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an expert at generating detailed, high-quality image descriptions for training vision-language models."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in GPT refinement: {str(e)}")
        return None

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    user_id = request.cookies.get('user_id')
    if not user_id:
        # Generate a temporary user_id if cookie is not present
        import uuid
        user_id = f"temp_{uuid.uuid4().hex[:8]}"
        print(f"Warning: No user_id cookie found, using temporary ID: {user_id}")

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_path = os.path.join(UPLOADS_DIR, f"recording_{timestamp}.wav")
    audio_file.save(audio_path)
    
    # Get image filename from form data
    image_filename = request.form.get('image_filename')
    
    # Load the full audio recording
    full_audio = AudioSegment.from_file(audio_path)
    
    # Get click timestamps from form data (in ms)
    click_timestamps_str = request.form.get('click_timestamps')
    click_timestamps = json.loads(click_timestamps_str) if click_timestamps_str else []
    
    # Define fixed duration to save before the first click (in ms)
    PRE_CLICK_DURATION_MS = 2000  # 2 seconds

    segmented_transcriptions = []
    
    # Determine segment start and end times
    segment_times = []  # List of (start_time_ms, end_time_ms)
    
    if not click_timestamps:
        segment_times.append((0, len(full_audio)))
    else:
        first_click_time = click_timestamps[0]
        start_time = max(0, first_click_time - PRE_CLICK_DURATION_MS)
        end_time = first_click_time
        segment_times.append((start_time, end_time))

        for i in range(len(click_timestamps) - 1):
            start_time = click_timestamps[i]
            end_time = click_timestamps[i+1]
            segment_times.append((start_time, end_time))

        last_click_time = click_timestamps[-1]
        start_time = last_click_time
        end_time = len(full_audio)
        segment_times.append((start_time, end_time))

    # Process each segment
    for i, (start_time, end_time) in enumerate(segment_times):
        if start_time >= end_time:
            continue  # Skip empty segments
            
        segment = full_audio[start_time:end_time]
        segment_path = os.path.join(UPLOADS_DIR, f"segment_{timestamp}_{i}.wav")
        segment.export(segment_path, format="wav")
        
        whisper_model = get_whisper_model()
        if not whisper_model:
            return jsonify({"error": "Whisper model not available"}), 503
        result = whisper_model.transcribe(segment_path)
        segment_transcription = result["text"]
        
        segmented_transcriptions.append({
            "segment_index": i,
            "start_time_ms": start_time,
            "end_time_ms": end_time,
            "transcription": segment_transcription.strip()
        })
        
        os.remove(segment_path)
    
    # Save segmented transcriptions
    segmented_transcription_path = os.path.join(TRANSCRIPTIONS_DIR, f"segmented_transcription_{timestamp}.json")
    with open(segmented_transcription_path, "w") as f:
        json.dump(segmented_transcriptions, f, indent=4)
    
    # Combine all transcriptions for GPT refinement
    combined_transcription = "\n".join([seg["transcription"] for seg in segmented_transcriptions])
    
    # Get OCR text if available
    ocr_text = request.form.get('ocr_text')
    
    # Get refined transcription from GPT with OCR text
    refined_transcription = refine_transcription_with_gpt(combined_transcription, image_filename, ocr_text)
    
    # Mark image as processed for the user
    image_filename = request.form.get('image_filename')
    if image_filename:
        processed_data = load_processed_images()
        if user_id not in processed_data:
            processed_data[user_id] = []
        if image_filename not in processed_data[user_id]:
            processed_data[user_id].append(image_filename)
        save_processed_images(processed_data)
    
    return jsonify({
        "segmented_transcriptions": segmented_transcriptions,
        "refined_transcription": refined_transcription,
        "timestamp": timestamp
    })

@app.route('/api/save-refined', methods=['POST'])
def save_refined():
    data = request.json
    timestamp = data.get('timestamp')
    refined_text = data.get('refined_text')
    
    if not timestamp or not refined_text:
        return jsonify({"error": "Missing required fields"}), 400
    
    refined_path = os.path.join(TRANSCRIPTIONS_DIR, f"refined_{timestamp}.txt")
    with open(refined_path, "w") as f:
        f.write(refined_text)
    
    return jsonify({"success": True})

@app.route('/api/save-objects', methods=['POST'])
def save_objects():
    data = request.json
    image_filename = data.get('image_filename')
    objects = data.get('objects', [])
    
    if not image_filename:
        return jsonify({"error": "Missing image filename"}), 400
    
    # Create a unique filename for the objects data
    base_filename = os.path.splitext(image_filename)[0]
    objects_dir = os.path.join(BASE_DIR, "static", "objects")
    os.makedirs(objects_dir, exist_ok=True)
    objects_path = os.path.join(objects_dir, f"{base_filename}_objects.json")
    
    # Save objects data
    with open(objects_path, 'w') as f:
        json.dump({
            "image_filename": image_filename,
            "objects": objects,
            "timestamp": datetime.now().isoformat()
        }, f, indent=4)
    
    return jsonify({"success": True})

@app.route('/api/get-objects/<image_filename>')
def get_objects(image_filename):
    base_filename = os.path.splitext(image_filename)[0]
    objects_path = os.path.join(BASE_DIR, "static", "objects", f"{base_filename}_objects.json")
    
    if os.path.exists(objects_path):
        with open(objects_path, 'r') as f:
            return jsonify(json.load(f))
    
    return jsonify({"objects": []})

@app.route('/api/export-objects', methods=['POST'])
def export_objects():
    try:
        data = request.json
        image_filename = data.get('image_filename')
        objects = data.get('objects')
        timestamp = data.get('timestamp')
        
        if not image_filename or not objects:
            return jsonify({"error": "Missing required fields"}), 400
        
        # Create export directory if it doesn't exist
        export_dir = os.path.join(BASE_DIR, "exports")
        os.makedirs(export_dir, exist_ok=True)
        
        # Create filename with timestamp
        base_filename = os.path.splitext(image_filename)[0]
        export_filename = f"{base_filename}_objects_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        export_path = os.path.join(export_dir, export_filename)
        
        # Export data with pretty formatting
        with open(export_path, 'w') as f:
            json.dump({
                "image_filename": image_filename,
                "timestamp": timestamp,
                "objects": objects
            }, f, indent=4)
        
        return jsonify({
            "success": True,
            "filename": export_filename
        })
        
    except Exception as e:
        print(f"Error exporting objects: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Note: The analyze_scientific_image function has been removed
# Scientific image analysis is now handled by the enhanced preprocessing pipeline
# Use /api/trigger-preprocessing/<task_id> to start processing
# Use /api/fetch-science-results/<image_id> to get results


@app.route('/api/trigger-preprocessing/<task_id>', methods=['POST'])
def trigger_preprocessing(task_id):
    """
    Trigger preprocessing for a specific task
    This endpoint is called after task creation to automatically process the task
    """
    try:
        # Import the preprocessing function
        from enhanced_preprocessing import process_task_async
        import threading
        
        # Start preprocessing in a background thread
        thread = threading.Thread(target=process_task_async, args=(task_id,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "success": True,
            "message": f"Preprocessing started for task {task_id}"
        })
        
    except Exception as e:
        print(f"Error triggering preprocessing for task {task_id}: {str(e)}")
        return jsonify({
            "error": f"Failed to trigger preprocessing: {str(e)}"
        }), 500
    
@app.route('/api/fetch-science-results/<image_id>', methods=['GET'])
def fetch_science_results(image_id):
    """
    Fetch preprocessed science analysis results for an image
    This endpoint is called by the frontend to load preprocessed results
    """
    try:
        from enhanced_preprocessing import fetch_preprocessed_science_results
        results = fetch_preprocessed_science_results(image_id)
        if results:
            return jsonify(results)
        else:
            return jsonify({
                "error": "No preprocessed results found for this image"
            }), 404
    except Exception as e:
        print(f"Error fetching science results for image {image_id}: {str(e)}")
        return jsonify({
            "error": f"Failed to fetch results: {str(e)}"
        }), 500

@app.route('/api/extract-pdf-images', methods=['POST'])
def extract_pdf_images():
    """Extract images from uploaded PDF file."""
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No PDF file provided'}), 400
        pdf_file = request.files['pdf']
        if pdf_file.filename == '':
            return jsonify({'error': 'No PDF file selected'}), 400
        if not pdf_file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'File must be a PDF'}), 400
        # Save PDF to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            pdf_file.save(temp_pdf.name)
            temp_pdf_path = temp_pdf.name
        try:
            # Extract images from PDF
            extracted_images = extract_images_from_pdf(temp_pdf_path)
            return jsonify({
                'success': True,
                'images': extracted_images,
                'count': len(extracted_images)
            })
        finally:
            # Clean up temporary PDF file
            if os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)
    except Exception as e:
        print(f"Error extracting PDF images: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/segregate-image', methods=['POST'])
def segregate_image_api():
    """Auto-segregate an uploaded image into subfigures using split_subfigures.py logic."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get optional parameters from request
        bg_tol = request.form.get('bg_tol', 30, type=int)
        gap_frac = request.form.get('gap_frac', 0.02, type=float)
        select_frac = request.form.get('select_frac', 0.5, type=float)
        min_panel_frac = request.form.get('min_panel_frac', 0.005, type=float)
        
        # Save image to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_img:
            image_file.save(temp_img.name)
            temp_img_path = temp_img.name
        
        # Create temp output directory
        temp_out_dir = tempfile.mkdtemp()
        
        # Import and use split_subfigures functions
        from split_subfigures import extract_matplotlib_subfigures
        
        try:
            # Call extract_matplotlib_subfigures with the provided parameters
            panels = extract_matplotlib_subfigures(
                temp_img_path,
                output_folder=temp_out_dir,
                bg_tol=bg_tol,
                gap_frac=gap_frac,
                select_frac=select_frac,
                min_panel_frac=min_panel_frac
            )
            
            # Collect all subfigure images
            subfigures = []
            for fname in sorted(os.listdir(temp_out_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    with open(os.path.join(temp_out_dir, fname), 'rb') as f:
                        img_bytes = f.read()
                        img_b64 = 'data:image/png;base64,' + base64.b64encode(img_bytes).decode('utf-8')
                        subfigures.append({'url': img_b64, 'filename': fname})
            
            # Clean up temp files
            os.remove(temp_img_path)
            import shutil
            shutil.rmtree(temp_out_dir)
            
            return jsonify({
                'subfigures': subfigures,
                'panel_count': len(panels),
                'parameters_used': {
                    'bg_tol': bg_tol,
                    'gap_frac': gap_frac,
                    'select_frac': select_frac,
                    'min_panel_frac': min_panel_frac
                }
            })
            
        except ValueError as ve:
            # Handle specific errors from split_subfigures
            error_msg = str(ve)
            if "No gutter segments found" in error_msg:
                return jsonify({
                    'error': 'No gutter segments found. Try adjusting parameters: lower gap_frac or increase bg_tol',
                    'suggestion': 'Try gap_frac=0.01 or bg_tol=50'
                }), 400
            elif "No panels left after min_panel_frac filtering" in error_msg:
                return jsonify({
                    'error': 'No panels detected after filtering. Try lowering min_panel_frac',
                    'suggestion': 'Try min_panel_frac=0.001'
                }), 400
        else:
                return jsonify({'error': error_msg}), 400
                
    except Exception as e:
        print(f"Error in segregate_image_api: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# MedSAM Endpoints (Using HuggingFace Space API)
@app.route('/api/medsam/load_from_supabase', methods=['POST'])
def medsam_load_from_supabase():
    """Load image and masks from Supabase database for segmentation via HuggingFace"""
    try:
        data = request.get_json()
        image_id = data.get('image_id')
        
        if not image_id:
            return jsonify({'error': 'image_id is required'}), 400

        # Import Supabase utilities
        from utils.supabase_client import SupabaseManager

        # Initialize components
        supabase_manager = SupabaseManager()
        
        # Get HuggingFace segmentation client
        segmentation_client = get_segmentation_client()

        print(f"Loading image and masks for image_id: {image_id}")
        
        # Get image and basic info from Supabase (without processing masks)
        data = supabase_manager.get_image_and_basic_info(image_id)

        # Load image for HuggingFace API (will be sent to the Space)
        success = segmentation_client.load_image(data['temp_image_path'])
        if not success:
            return jsonify({'error': 'Failed to load image for segmentation'}), 500

        # Get mask count without processing masks
        mask_count = supabase_manager.get_mask_count(image_id)

        # Create a minimal response to avoid JSON size issues
        response_data = {
            'success': True,
            'image_url': data['signed_url'],
            'image_data': data['image_data'],
            'mask_count': mask_count,
            'temp_image_path': data['temp_image_path']
        }

        print(f"Successfully loaded image with {mask_count} masks from Supabase")
        print(f"Segmentation will use HuggingFace Space: {HUGGINGFACE_SPACE_URL}")
        
        try:
            response = jsonify(response_data)
            return response
        except Exception as json_error:
            print(f"JSON serialization error: {json_error}")
            import traceback
            traceback.print_exc()
            
            # Fallback: return minimal response
            fallback_response = {
                'success': True,
                'image_url': data['signed_url'],
                'mask_count': mask_count,
                'error': 'Masks too large for JSON response'
            }
            print(f"Returning fallback response: {fallback_response}")
            return jsonify(fallback_response)
        
    except Exception as e:
        print(f"Error loading from Supabase: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to load from Supabase: {str(e)}'}), 500

# Global cache for masks to avoid reloading
masks_cache = {}

@app.route('/api/medsam/get_masks', methods=['POST'])
def medsam_get_masks():
    """Get masks for an image with pagination - returns metadata only"""
    try:
        data = request.get_json()
        image_id = data.get('image_id')
        page = data.get('page', 0)
        page_size = data.get('page_size', 10)
        
        if not image_id:
            return jsonify({'error': 'image_id is required'}), 400
        
        print(f"Getting masks metadata for image {image_id}, page {page}, page_size {page_size}")
        
        # Import Supabase manager
        from utils.supabase_client import SupabaseManager
        supabase_manager = SupabaseManager()
        
        # Get masks metadata from database (just the metadata, not the actual mask images)
        try:
            masks_metadata = supabase_manager.get_masks_for_image(str(image_id))
            print(f"Found {len(masks_metadata)} masks in database")
        except Exception as db_error:
            print(f"Database error: {db_error}")
            # Return empty result if database query fails
            return jsonify({
                'success': True,
                'masks': [],
                'total_masks': 0,
                'page': page,
                'page_size': page_size,
                'has_more': False
            })
        
        # Apply pagination
        start_idx = page * page_size
        end_idx = start_idx + page_size
        paginated_masks = masks_metadata[start_idx:end_idx]
        
        print(f"Returning {len(paginated_masks)} masks for page {page}")
        
        response_data = {
            'success': True,
            'masks': paginated_masks,
            'total_masks': len(masks_metadata),
            'page': page,
            'page_size': page_size,
            'has_more': end_idx < len(masks_metadata)
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error getting masks: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return a valid JSON response even on error
        return jsonify({
            'success': False,
            'error': f'Failed to get masks: {str(e)}',
            'masks': [],
            'total_masks': 0,
            'page': page,
            'page_size': page_size,
            'has_more': False
        }), 500

@app.route('/api/medsam/get_mask_image', methods=['POST'])
def medsam_get_mask_image():
    """Download and convert a mask image from storage to array"""
    try:
        data = request.get_json()
        mask_url = data.get('mask_url')
        
        if not mask_url:
            return jsonify({'error': 'mask_url is required'}), 400
        
        print(f"Downloading mask from: {mask_url}")
        
        # Import Supabase manager
        from utils.supabase_client import SupabaseManager
        supabase_manager = SupabaseManager()
        
        # Get signed URL for the mask
        try:
            from config import MASKS_BUCKET
            signed_url = supabase_manager.get_signed_url(mask_url, bucket=MASKS_BUCKET)
            print(f"Got signed URL: {signed_url}")
            
            # Download and convert mask
            mask_array = supabase_manager.download_mask_to_array(signed_url)
            
            # Convert to list for JSON serialization
            mask_list = mask_array.tolist()
            
            return jsonify({
                'success': True,
                'mask_array': mask_list,
                'shape': mask_array.shape
            })
            
        except Exception as e:
            print(f"Error processing mask: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Failed to process mask: {str(e)}'}), 500
        
    except Exception as e:
        print(f"Error getting mask image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to get mask image: {str(e)}'}), 500

@app.route('/api/medsam/clear_cache', methods=['POST'])
def medsam_clear_cache():
    """Clear the masks cache"""
    try:
        global masks_cache
        masks_cache.clear()
        print("Masks cache cleared")
        return jsonify({'success': True, 'message': 'Cache cleared'})
    except Exception as e:
        return jsonify({'error': f'Failed to clear cache: {str(e)}'}), 500

@app.route('/api/medsam/segment_points', methods=['POST'])
def medsam_segment_points():
    """Segment using points via HuggingFace Space API"""
    try:
        data = request.get_json()
        points = data.get('points', [])  # [[x1, y1], [x2, y2], ...]
        labels = data.get('labels', [])  # [1, 0, 1, ...] (1=foreground, 0=background)

        print(f"Received points: {points}")
        print(f"Received labels: {labels}")

        if not points:
            return jsonify({'error': 'At least one point is required'}), 400

        # Get HuggingFace segmentation client
        segmentation_client = get_segmentation_client()
        
        if segmentation_client.current_image is None:
            return jsonify({'error': 'No image loaded. Please load an image first.'}), 500

        print(f"Calling HuggingFace Space for {len(points)} points segmentation")
        
        # Call HuggingFace API
        result = segmentation_client.segment_with_points(points, labels)
        
        if result:
            return jsonify({
                'success': True,
                'masks': result['masks'],
                'confidences': result['confidences'],
                'method': result.get('method', 'huggingface_points')
            })
        else:
            return jsonify({'error': 'Segmentation failed'}), 500
        
    except Exception as e:
        print(f"Error in point segmentation: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Segmentation failed: {str(e)}'}), 500

@app.route('/api/medsam/segment_box', methods=['POST'])
def medsam_segment_box():
    """Segment using bounding box via HuggingFace Space API"""
    try:
        data = request.get_json()
        bbox = data.get('bbox', [])  # [x1, y1, x2, y2]
        
        print(f"Received bbox: {bbox}")
        
        if not bbox or len(bbox) != 4:
            return jsonify({'error': 'Valid bounding box is required'}), 400
        
        # Get HuggingFace segmentation client
        segmentation_client = get_segmentation_client()
        
        if segmentation_client.current_image is None:
            return jsonify({'error': 'No image loaded. Please load an image first.'}), 500
        
        print(f"Calling HuggingFace Space for box segmentation")
        
        # Call HuggingFace API
        result = segmentation_client.segment_with_box(bbox)
        
        if result:
            print(f"Segmentation successful")
            
            return jsonify({
                'success': True,
                'mask': result['mask'],
                'confidence': result['confidence'],
                'method': result.get('method', 'huggingface_box')
            })
        else:
            print("Segmentation returned None")
            return jsonify({'error': 'Segmentation failed'}), 500
        
    except Exception as e:
        print(f"Error in box segmentation: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Segmentation failed: {str(e)}'}), 500

@app.route('/api/medsam/segment_combined', methods=['POST'])
def medsam_segment_combined():
    """Segment using both points and bounding box via HuggingFace Space API"""
    try:
        data = request.get_json()
        points = data.get('points', [])
        labels = data.get('labels', [])
        bbox = data.get('bbox', [])
        
        if not points or not labels or not bbox:
            return jsonify({'error': 'Points, labels, and bounding box are required'}), 400
        
        # Get HuggingFace segmentation client
        segmentation_client = get_segmentation_client()
        
        if segmentation_client.current_image is None:
            return jsonify({'error': 'No image loaded. Please load an image first.'}), 500

        print(f"Calling HuggingFace Space for combined segmentation")
        
        # Call HuggingFace API (uses box when both are provided)
        result = segmentation_client.segment_with_points_and_box(points, labels, bbox)
        
        if result:
            return jsonify({
                'success': True,
                'mask': result['mask'],
                'confidence': result['confidence'],
                'method': result.get('method', 'huggingface_combined')
            })
        else:
            return jsonify({'error': 'Segmentation failed'}), 500
            
    except Exception as e:
        print(f"Error in combined segmentation: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Segmentation failed: {str(e)}'}), 500

@app.route('/api/medsam/segment_multiple_boxes', methods=['POST'])
def medsam_segment_multiple_boxes():
    """Segment using multiple bounding boxes via HuggingFace Space API"""
    try:
        data = request.get_json()
        bboxes = data.get('bboxes', [])
        
        if not bboxes:
            return jsonify({'error': 'At least one bounding box is required'}), 400
        
        # Get HuggingFace segmentation client
        segmentation_client = get_segmentation_client()
        
        if segmentation_client.current_image is None:
            return jsonify({'error': 'No image loaded. Please load an image first.'}), 500
        
        print(f"Calling HuggingFace Space for {len(bboxes)} boxes segmentation")
        
        # Call HuggingFace API for all boxes at once
        result = segmentation_client.segment_with_multiple_boxes(bboxes)

        if result:
            return jsonify({
                'success': True,
                'masks': result['masks'],
                'confidences': result['confidences'],
                'method': result.get('method', 'huggingface_multiple_boxes')
            })
        else:
            return jsonify({'error': 'Segmentation failed'}), 500
            
    except Exception as e:
        print(f"Error in multiple box segmentation: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Segmentation failed: {str(e)}'}), 500

@app.route('/api/medsam/status')
def medsam_status():
    """Check segmentation service availability (HuggingFace Space)"""
    try:
        segmentation_client = get_segmentation_client()
        
        return jsonify({
            'available': segmentation_client.is_available(),
            'initialized': True,
            'device': 'huggingface_cloud',
            'image_loaded': segmentation_client.current_image is not None,
            'embedding_loaded': True,  # HuggingFace handles embeddings server-side
            'backend': 'huggingface_space',
            'space_url': HUGGINGFACE_SPACE_URL
        })
    except Exception as e:
        return jsonify({
            'available': False,
            'initialized': False,
            'error': str(e),
            'backend': 'huggingface_space'
        })

@app.route('/api/medsam/save_manual_masks', methods=['POST'])
def medsam_save_manual_masks():
    """Save manually created masks to database using the preprocessing pipeline"""
    try:
        data = request.get_json()
        image_id = data.get('image_id')
        masks_data = data.get('masks', [])
        
        if not image_id:
            return jsonify({'error': 'image_id is required'}), 400
        
        if not masks_data or len(masks_data) == 0:
            return jsonify({'success': True, 'message': 'No masks to save', 'saved_count': 0})
        
        print(f"Saving {len(masks_data)} manually created masks for image {image_id}")
        
        # Import the save_masks function from enhanced_preprocessing
        import sys
        import os
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)
        
        from enhanced_preprocessing import save_masks
        import numpy as np
        
        # Convert frontend mask data to SAM format
        formatted_masks = []
        for idx, mask_data in enumerate(masks_data):
            try:
                # Extract mask array
                if isinstance(mask_data, dict):
                    mask_array = np.array(mask_data.get('mask', mask_data))
                    bbox = mask_data.get('bbox', [0, 0, 0, 0])
                    confidence = mask_data.get('confidence', 0.9)
                    points = mask_data.get('points', [])
                    method = mask_data.get('method', 'manual')
                else:
                    mask_array = np.array(mask_data)
                    bbox = [0, 0, mask_array.shape[1], mask_array.shape[0]]
                    confidence = 0.9
                    points = []
                    method = 'manual'
                
                # Ensure mask is boolean
                if mask_array.dtype != bool:
                    mask_array = mask_array.astype(bool)
                
                # Calculate area
                area = int(np.sum(mask_array))
                
                # Format in SAM-compatible structure
                formatted_mask = {
                    'segmentation': mask_array,
                    'area': area,
                    'bbox': bbox if isinstance(bbox, list) else [0, 0, 0, 0],
                    'predicted_iou': confidence,
                    'point_coords': points if isinstance(points, list) else [],
                    'stability_score': confidence,
                    'crop_box': bbox if isinstance(bbox, list) else [0, 0, 0, 0],
                    'method': method  # Track that this is manual
                }
                
                formatted_masks.append(formatted_mask)
                print(f"Formatted mask {idx + 1}/{len(masks_data)}: area={area}, bbox={bbox}")
                
            except Exception as e:
                print(f"Error formatting mask {idx}: {e}")
                continue
        
        if len(formatted_masks) == 0:
            return jsonify({'error': 'No valid masks to save'}), 400
        
        # Use the existing save_masks function
        try:
            save_masks(str(image_id), formatted_masks)
            print(f"Successfully saved {len(formatted_masks)} masks to database")
            
            return jsonify({
                'success': True,
                'message': f'Successfully saved {len(formatted_masks)} masks',
                'saved_count': len(formatted_masks)
            })
            
        except Exception as save_error:
            print(f"Error in save_masks function: {save_error}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Failed to save masks: {str(save_error)}'}), 500
        
    except Exception as e:
        print(f"Error saving manual masks: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to save manual masks: {str(e)}'}), 500

if __name__ == '__main__':
    # You can customize the host and port here
    # app.run(debug=True)  # Default: 127.0.0.1:5000
    # app.run(debug=True, host='0.0.0.0', port=5000)  # Allow external connections
    # app.run(debug=True, host='localhost', port=5000)  # Custom port
    app.run(debug=True, host='127.0.0.1', port=5000) 