import os
import io
import uuid
import time
import requests
import numpy as np
import cv2
import json
import base64
from urllib.parse import urlparse
from skimage import transform
from PIL import Image
from supabase import create_client, Client
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import gradio_client for HuggingFace API
# IMPORTANT: Remove proxy env vars BEFORE importing to avoid proxy parameter errors
# gradio-client 0.7.0 reads these but doesn't support proxy parameter
_proxy_vars_backup = {}
_proxy_vars_to_remove = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
for var in _proxy_vars_to_remove:
    if var in os.environ:
        _proxy_vars_backup[var] = os.environ.pop(var)

try:
    from gradio_client import Client as GradioClient
    try:
        from gradio_client import handle_file
    except ImportError:
        # handle_file might not exist in older versions, define a fallback
        def handle_file(file_path):
            return file_path
    GRADIO_CLIENT_AVAILABLE = True
    logger.debug("gradio_client imported successfully")
except ImportError as e:
    logger.warning(f"gradio_client not installed. Install with: pip install gradio-client")
    logger.warning(f"Import error details: {e}")
    GRADIO_CLIENT_AVAILABLE = False
    GradioClient = None
    handle_file = None
finally:
    # Restore proxy vars after import check
    for var, value in _proxy_vars_backup.items():
        os.environ[var] = value
    _proxy_vars_backup = {}

# ─── CONFIG ──────────────────────────────────────────────────────────────
SUPABASE_URL       = os.getenv("REACT_APP_SUPABASE_URL")
SUPABASE_KEY       = os.getenv("REACT_APP_SUPABASE_ANON_KEY")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Enhanced 21-class categories for comprehensive chart element detection
ENHANCED_CLASS_NAMES = [
    'title', 'subtitle', 'x-axis', 'y-axis', 'x-axis-label', 'y-axis-label',
    'x-tick-label', 'y-tick-label', 'legend', 'legend-title', 'legend-item',
    'data-point', 'data-line', 'data-bar', 'data-area', 'grid-line',
    'axis-title', 'tick-label', 'data-label', 'legend-text', 'plot-area'
]

# Database tables
TASKS_TABLE        = "tasks"
CATEGORIES_TABLE   = "categories"
IMAGES_TABLE       = "images"
MASK_BUCKET        = "masks"
EMBED_BUCKET       = "embeddings"
MASKS_TABLE        = "masks2"
EMBED_TABLE        = "embeddings2"
IMAGE_BUCKET       = "images"
CHART_ANALYSIS_TABLE = "chart_analysis"
CHART_ELEMENTS_TABLE = "chart_elements"
TEXT_ELEMENTS_TABLE = "text_elements"

# ─── CLIENT INIT ───────────────────────────────────────────────────────────
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─── HUGGINGFACE API SETUP (PRIMARY) ──────────────────────────────────────
# HuggingFace Space for SAM/MedSAM inference
MEDSAM_HF_SPACE = os.getenv("MEDSAM_HF_SPACE", "Aniketg6/medsam-inference")
HF_SPACE_URL = "https://hanszhu-dense-captioning-platform.hf.space"

# Initialize HuggingFace client for auto mask generation
hf_mask_client = None
USE_HF_FOR_MASKS = os.getenv("USE_HF_FOR_MASKS", "true").lower() == "true"

if GRADIO_CLIENT_AVAILABLE and USE_HF_FOR_MASKS:
    # IMPORTANT: Remove proxy env vars BEFORE initializing Client
    # gradio-client 0.7.0 reads these but doesn't support proxy parameter
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy']
    saved_proxy_vars_init = {}
    for var in proxy_vars:
        if var in os.environ:
            saved_proxy_vars_init[var] = os.environ.pop(var)
    
    try:
        # Monkey-patch GradioClient.__init__ to remove proxy parameter if present
        original_gradio_init = GradioClient.__init__
        def patched_gradio_init(self, *args, **kwargs):
            # Remove proxy-related kwargs that gradio-client 0.7.0 doesn't support
            kwargs.pop('proxy', None)
            kwargs.pop('proxies', None)
            # Call original init
            return original_gradio_init(self, *args, **kwargs)
        GradioClient.__init__ = patched_gradio_init
        
        logger.info(f"Connecting to HuggingFace Space: {MEDSAM_HF_SPACE}")
        # Get HuggingFace token from environment variable
        hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
        
        # Double-check proxy vars are still removed right before Client creation
        for var in proxy_vars:
            if var in os.environ:
                logger.warning(f"Proxy var {var} still present, removing again")
                saved_proxy_vars_init[var] = os.environ.pop(var)
        
        # Initialize client - proxy env vars already removed above
        try:
            if hf_token:
                logger.info("Using HuggingFace token for authentication")
                hf_mask_client = GradioClient(MEDSAM_HF_SPACE, hf_token=hf_token)
            else:
                logger.info("No HF_TOKEN found - connecting without authentication")
                hf_mask_client = GradioClient(MEDSAM_HF_SPACE)
        except (TypeError, ValueError) as te:
            error_msg = str(te).lower()
            if "proxy" in error_msg or "unexpected keyword" in error_msg:
                logger.error(f"Client initialization failed with proxy error: {te}")
                logger.error("gradio-client 0.7.0 doesn't support proxy parameter")
                logger.error("HuggingFace mask generation will be disabled")
                hf_mask_client = None
            else:
                raise
        
        # Check if auto mask generation is available
        try:
            status_result = hf_mask_client.predict(api_name="/check_auto_mask_status")
            status = json.loads(status_result)
            if status.get('available'):
                logger.info(f"✓ HuggingFace auto mask generation available (device: {status.get('device')})")
            else:
                logger.warning("HuggingFace Space connected but SAM-H model not loaded")
                logger.warning("Automatic mask generation will use local model if available")
        except Exception as e:
            logger.warning(f"Could not check HuggingFace auto mask status: {e}")
            
    except json.JSONDecodeError as je:
        logger.warning(f"Could not connect to HuggingFace Space: JSON decode error")
        logger.warning(f"Error: {je}")
        logger.warning(f"This usually means the Space is down, returned HTML instead of JSON, or has network issues")
        logger.warning(f"Space URL: {MEDSAM_HF_SPACE}")
        logger.warning(f"💡 Check if the Space is running at https://huggingface.co/spaces/{MEDSAM_HF_SPACE}")
        hf_mask_client = None
    except Exception as e:
        logger.warning(f"Could not connect to HuggingFace Space: {e}")
        logger.warning(f"Error details: {type(e).__name__}: {str(e)}")
        hf_mask_client = None
    finally:
        # Restore proxy environment variables
        for var, value in saved_proxy_vars_init.items():
            os.environ[var] = value

# ─── HF MASK GENERATION API ────────────────────────────────────────────────

def generate_masks_via_huggingface(image_path: str) -> list:
    """
    Generate automatic masks using HuggingFace Space API
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of masks in the same format as SamAutomaticMaskGenerator.generate()
    """
    if not hf_mask_client:
        logger.warning("HuggingFace mask client not available")
        return None
    
    try:
        logger.info(f"Calling HuggingFace API for automatic mask generation...")

        # Use the same parameter strategy as test_auto.py:
        # - max_masks = -1 => return all masks
        # - resize_longest = 512 => downscale long side for speed / smaller payload
        params = {"max_masks": -1, "resize_longest": 512}

        start = time.perf_counter()
        result = hf_mask_client.predict(
            image=handle_file(image_path),
            request_json=json.dumps(params),
            api_name="/generate_auto_masks",
        )
        elapsed = time.perf_counter() - start

        # Parse the result
        data = json.loads(result)
        logger.info(
            f"HuggingFace auto mask API call completed in {elapsed:.2f}s "
            f"(success={data.get('success')}, num_masks={data.get('num_masks')})"
        )
        
        if not data.get('success'):
            logger.error(f"HuggingFace API error: {data.get('error')}")
            return None
        
        masks = data.get('masks', [])
        logger.info(f"HuggingFace API generated {len(masks)} masks")
        
        # Convert back to numpy format expected by save_masks
        processed_masks = []
        for m in masks:
            processed_mask = {
                'segmentation': np.array(m['segmentation'], dtype=bool),
                'area': m['area'],
                'bbox': m['bbox'],
                'predicted_iou': m['predicted_iou'],
                'point_coords': np.array(m['point_coords']) if m['point_coords'] else None,
                'stability_score': m['stability_score'],
                'crop_box': m['crop_box']
            }
            processed_masks.append(processed_mask)
        
        return processed_masks
        
    except Exception as e:
        logger.error(f"Error generating masks via HuggingFace: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def generate_embedding_via_huggingface(image_path: str, image_id: str) -> bool:
    """
    Generate an image embedding using HuggingFace Space encode_image API
    and save it to Supabase (embeddings2 table + embeddings bucket).

    Args:
        image_path: Path to the image file
        image_id:   Supabase image_id (UUID string)

    Returns:
        True on success, False otherwise
    """
    if not hf_mask_client:
        logger.warning("HuggingFace mask client not available (cannot generate embeddings)")
        return False

    try:
        logger.info(f"Calling HuggingFace encode_image API for image_id={image_id}...")

        params = {"image_id": image_id}

        start = time.perf_counter()
        result = hf_mask_client.predict(
            image=handle_file(image_path),
            request_json=json.dumps(params),
            api_name="/encode_image",
        )
        elapsed = time.perf_counter() - start

        data = json.loads(result)
        if not data.get("success"):
            logger.error(f"HuggingFace encode_image error: {data.get('error')}")
            if "traceback" in data:
                logger.error(data["traceback"])
            return False

        embedding_b64 = data.get("embedding_npy_base64")
        embedding_shape = data.get("embedding_shape")

        if not embedding_b64:
            logger.error("encode_image API returned no embedding_npy_base64")
            return False

        logger.info(
            f"encode_image completed in {elapsed:.2f}s for image_id={image_id}, "
            f"embedding_shape={embedding_shape}"
        )

        # Decode base64 .npy -> numpy array of shape [C, H, W]
        try:
            embedding_bytes = base64.b64decode(embedding_b64)
            arr = np.load(io.BytesIO(embedding_bytes))
        except Exception as dec_err:
            logger.error(f"Failed to decode embedding from encode_image: {dec_err}")
            return False

        # Save embedding array to Supabase
        try:
            path = upload_embedding(image_id, arr)
            supabase.table(EMBED_TABLE).insert(
                {"image_id": image_id, "file_path": path}
            ).execute()
            logger.info(
                f"✓ Saved embedding for image_id={image_id} to Supabase at {path}"
            )
            return True
        except Exception as save_err:
            logger.error(f"Failed to save embedding to Supabase: {save_err}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    except Exception as e:
        logger.error(f"Error generating embedding via HuggingFace: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def generate_masks(img_np: np.ndarray, temp_image_path: str = None) -> list:
    """
    Generate automatic masks using HuggingFace Space only.
    
    Args:
        img_np: Image as numpy array
        temp_image_path: Optional path to temp image file (needed for HuggingFace API)
        
    Returns:
        List of masks
    """
    # Only HuggingFace Space is used now; no local SAM fallback.
    if hf_mask_client and temp_image_path and USE_HF_FOR_MASKS:
        masks = generate_masks_via_huggingface(temp_image_path)
        if masks:
            logger.info("Using masks from HuggingFace API")
            return masks
        else:
            logger.error("HuggingFace API failed to generate masks.")
            return None

    logger.error("HuggingFace mask client not available or temp_image_path missing.")
    return None


# ─── HF SPACE API FUNCTIONS ─────────────────────────────────────────────────

def perform_local_ocr(img_np: np.ndarray):
    """
    Perform local OCR using EasyOCR for text detection
    Returns list of text elements with bbox and confidence
    """
    try:
        import easyocr
        
        # Initialize EasyOCR reader if not already done
        if not hasattr(perform_local_ocr, 'reader'):
            perform_local_ocr.reader = easyocr.Reader(['en'])
        
        logger.info("Running local EasyOCR for text detection")
        
        # Perform OCR
        results = perform_local_ocr.reader.readtext(img_np)
        
        text_elements = []
        for (bbox, text, confidence) in results:
            # Convert bbox format from EasyOCR to our format
            # EasyOCR bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            # Our format is [x1, y1, x2, y2]
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            bbox_formatted = [
                float(min(x_coords)),  # x1
                float(min(y_coords)),  # y1
                float(max(x_coords)),  # x2
                float(max(y_coords))   # y2
            ]
            
            text_elements.append({
                'text': text,
                'bbox': bbox_formatted,
                'confidence': confidence,
                'element_type': 'text',
                'label': 'text'
            })
        
        logger.info(f"Local OCR detected {len(text_elements)} text elements")
        return text_elements
        
    except Exception as e:
        logger.error(f"Error in local OCR: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return []

def call_hf_space_api(image_path):
    """
    Call the Dense Captioning Platform API using gradio_client to analyze a scientific image
    Returns the raw API response
    """
    # IMPORTANT: Remove proxy env vars BEFORE importing gradio_client
    # gradio-client 0.7.0 reads these but doesn't support proxy parameter
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy']
    saved_proxy_vars = {}
    for var in proxy_vars:
        if var in os.environ:
            saved_proxy_vars[var] = os.environ.pop(var)
            logger.debug(f"Temporarily removed {var} to avoid gradio-client proxy error")
    
    try:
        # Check if gradio_client is available
        try:
            from gradio_client import Client, handle_file
        except ImportError:
            logger.error("gradio_client not available - cannot call HF Space API")
            return None
        
        # Monkey-patch Client.__init__ to remove proxy parameter if present
        # This prevents "unexpected keyword argument 'proxy'" errors
        original_init = Client.__init__
        def patched_init(self, *args, **kwargs):
            # Remove proxy-related kwargs that gradio-client 0.7.0 doesn't support
            kwargs.pop('proxy', None)
            kwargs.pop('proxies', None)
            # Call original init
            return original_init(self, *args, **kwargs)
        Client.__init__ = patched_init
        
        logger.info(f"Calling Dense Captioning Platform API: {HF_SPACE_URL}")
        
        # Get HuggingFace token from environment variable
        hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
        
        # Double-check proxy vars are still removed right before Client creation
        for var in proxy_vars:
            if var in os.environ:
                logger.warning(f"Proxy var {var} still present, removing again")
                saved_proxy_vars[var] = os.environ.pop(var)
        
        # Initialize client with direct URL
        # gradio-client 0.7.0 doesn't support proxy parameter, so we removed proxy env vars above
        try:
            if hf_token:
                client = Client("https://hanszhu-dense-captioning-platform.hf.space", hf_token=hf_token)
            else:
                client = Client("https://hanszhu-dense-captioning-platform.hf.space")
        except (TypeError, ValueError) as te:
            error_msg = str(te).lower()
            if "proxy" in error_msg or "unexpected keyword" in error_msg:
                logger.error(f"Client initialization failed with proxy error: {te}")
                logger.error("gradio-client 0.7.0 doesn't support proxy parameter")
                logger.error("Skipping HF Space API call - this is a gradio-client version limitation")
                return None
            else:
                raise
        
        # Call the predict function using the working approach
        try:
            result = client.predict(
                image=handle_file(image_path),
                fn_index=0
            )
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise e
        
        logger.info(f"API result received: {type(result)}")
        logger.info(f"API result content: {result}")
        return result
            
    except Exception as e:
        logger.error(f"Error calling Dense Captioning Platform API: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None
    finally:
        # Always restore proxy environment variables
        for var, value in saved_proxy_vars.items():
            os.environ[var] = value

def parse_hf_space_response(hf_response):
    """
    Parse the HF Space API response from gradio_client and convert it to the format expected by the pipeline
    """
    if not hf_response:
        logger.warning("HF Space response is empty or None")
        return None
    
    logger.info(f"Parsing HF Space response of type: {type(hf_response)}")
    
    try:
        # The gradio_client response might be a string or dict
        if isinstance(hf_response, str):
            # Check if it's an empty string
            if not hf_response.strip():
                logger.warning("HF Space response is empty string")
                return None
                
            # Try to parse as JSON if it's a string
            try:
                import json
                hf_response = json.loads(hf_response)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse as JSON: {e}")
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
                if 'data-bar' in element_type or 'bar' in element_type:
                    threshold = databar_threshold  # Very low for bars
                elif 'data-point' in element_type or 'data-line' in element_type:
                    threshold = datapoint_threshold  # Higher for data points/lines
                else:
                    threshold = element_threshold  # General elements
                
                if confidence >= threshold:
                    filtered_chart_elements.append(elem)
            
            # Filter data points by confidence and add them to chart elements
            filtered_data_points = []
            for dp in data_points:
                element_type = dp.get('element_type', '').lower()
                confidence = dp.get('confidence', 0)
                
                # Use different thresholds based on element type
                if 'data-bar' in element_type or 'bar' in element_type:
                    threshold = databar_threshold  # Very low for bars
                elif 'data-point' in element_type or 'data-line' in element_type:
                    threshold = datapoint_threshold  # Higher for data points/lines
                else:
                    threshold = element_threshold  # General elements
                
                if confidence >= threshold:
                    filtered_data_points.append(dp)
            
            # Combine chart elements and data points into a single list
            all_chart_elements = filtered_chart_elements + filtered_data_points
            
            # Group elements by type and add numbering
            element_type_groups = {}
            for elem in all_chart_elements:
                element_type = elem.get('element_type', '')
                if element_type not in element_type_groups:
                    element_type_groups[element_type] = []
                element_type_groups[element_type].append(elem)
            
            # Reconstruct list with elements grouped by type and numbered
            all_chart_elements = []
            for element_type, elements in element_type_groups.items():
                for i, elem in enumerate(elements, 1):
                    elem['element_type'] = f"{element_type} {i}"
                    all_chart_elements.append(elem)
            
            logger.info(f"Threshold filtering results:")
            logger.info(f"  Chart elements: {len(chart_elements)} -> {len(filtered_chart_elements)} (threshold: {databar_threshold} for bars, {element_threshold} for others)")
            logger.info(f"  Data points: {len(data_points)} -> {len(filtered_data_points)} (threshold: {datapoint_threshold})")
            logger.info(f"  Combined chart elements: {len(all_chart_elements)}")
            
            # Combine all results (text elements will be added by local OCR)
            analysis_results = {
                'chart_type': chart_type_name,
                'chart_type_confidence': 0.9 if chart_type_name else None,
                'chart_elements': all_chart_elements,
                'text_elements': [],  # Will be filled by local OCR
                'status': hf_response.get('status', 'unknown'),
                'processing_time': hf_response.get('processing_time', 0)
            }
            
            logger.info(f"Final parsed analysis results: chart_type={chart_type_name}, total elements={len(all_chart_elements)}")
            
            return analysis_results
        
    except Exception as e:
        logger.error(f"Error parsing HF Space response: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")


def parse_det_data_sample_string(det_data_sample_str, element_type_prefix="element"):
    """
    Parse a DetDataSample string format and convert it to the format expected by the pipeline
    
    Args:
        det_data_sample_str: The DetDataSample string to parse
        element_type_prefix: Prefix for element types ("chart_element" or "data_point")
    """
    try:
        if not det_data_sample_str or not isinstance(det_data_sample_str, str):
            return []
        
        import re
        
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
            # Split by individual bbox arrays - more robust pattern
            bbox_array_pattern = r'\[\s*([\d\.,\s\-]+)\s*\]'
            bbox_arrays = re.findall(bbox_array_pattern, bbox_content)
            
            logger.info(f"Found {len(bbox_arrays)} bbox arrays in content")
            
            for bbox_str in bbox_arrays:
                bbox_str = bbox_str.strip()
                if bbox_str:
                    try:
                        # Split by comma and convert to floats, handle whitespace better
                        bbox_values = [float(x.strip()) for x in bbox_str.split(',') if x.strip()]
                        if len(bbox_values) == 4:
                            bboxes.append(bbox_values)
                        else:
                            logger.warning(f"Invalid bbox format: {bbox_str} (expected 4 values, got {len(bbox_values)})")
                    except ValueError as e:
                        logger.warning(f"Failed to parse bbox values from '{bbox_str}': {e}")
                        continue
        
        # Parse labels - look for tensor format
        label_pattern = r'labels:\s*tensor\(\[([\d\s,]+)\]\)'
        label_matches = re.findall(label_pattern, det_data_sample_str)
        if label_matches:
            label_str = label_matches[0]
            labels = [int(x.strip()) for x in label_str.split(',') if x.strip()]
            logger.info(f"Found {len(labels)} labels")
        else:
            logger.warning("No labels found in DetDataSample string")
        
        # Parse scores - look for tensor format
        score_pattern = r'scores:\s*tensor\(\[([\d\.\s,]+)\]\)'
        score_matches = re.findall(score_pattern, det_data_sample_str)
        if score_matches:
            score_str = score_matches[0]
            scores = [float(x.strip()) for x in score_str.split(',') if x.strip()]
            logger.info(f"Found {len(scores)} scores")
        else:
            logger.warning("No scores found in DetDataSample string")
        
        # Create elements with proper type prefix
        elements = []
        min_length = min(len(bboxes), len(labels), len(scores))
        logger.info(f"Creating elements: bboxes={len(bboxes)}, labels={len(labels)}, scores={len(scores)}, min_length={min_length}")
        
        for i in range(min_length):
            label_idx = labels[i]
            # Use meaningful class names from ENHANCED_CLASS_NAMES
            if label_idx < len(ENHANCED_CLASS_NAMES):
                element_type = ENHANCED_CLASS_NAMES[label_idx]
            else:
                element_type = f'{element_type_prefix}_{label_idx}'
            elements.append({
                'element_type': element_type,
                'bbox': bboxes[i],
                'confidence': scores[i],
                'label': element_type
            })
        
        logger.info(f"Parsed {len(elements)} {element_type_prefix} elements from DetDataSample string")
        return elements
        
    except Exception as e:
        logger.error(f"Error parsing DetDataSample string: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return []

# ─── FETCH FUNCTIONS ────────────────────────────────────────────────────────

def fetch_chart_analysis(image_id: str):
    """Fetch chart type classification results for an image"""
    try:
        response = supabase.table(CHART_ANALYSIS_TABLE).select('*').eq('image_id', image_id).execute()
        if response.data:
            return response.data[0]  # Return the first (and should be only) result
        return None
    except Exception as e:
        logger.error(f"Error fetching chart analysis for image {image_id}: {e}")
        return None

def fetch_chart_elements(image_id: str):
    """Fetch chart element detection results for an image (including data points)"""
    try:
        response = supabase.table(CHART_ELEMENTS_TABLE).select('*').eq('image_id', image_id).execute()
        logger.info(f"Fetched {len(response.data)} chart elements (including data points) from database for image {image_id}")
        
        elements = []
        for record in response.data:
            # Convert database format back to analysis format
            element = {
                'element_type': record['element_type'],
                'confidence': float(record['confidence']),
                'bbox': [
                    float(record['bbox_x']),
                    float(record['bbox_y']),
                    float(record['bbox_x'] + record['bbox_width']),
                    float(record['bbox_y'] + record['bbox_height'])
                ]
            }
            elements.append(element)
        
        # Log confidence distribution of fetched elements
        if elements:
            confidences = [elem['confidence'] for elem in elements]
            logger.info(f"Fetched chart element confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
            logger.info(f"Fetched chart elements with confidence >= 0.4: {sum(1 for c in confidences if c >= 0.4)}")
            logger.info(f"Fetched chart elements with confidence >= 0.1: {sum(1 for c in confidences if c >= 0.1)}")
        
        return elements
    except Exception as e:
        logger.error(f"Error fetching chart elements for image {image_id}: {e}")
        return []



def fetch_text_elements(image_id: str):
    """Fetch OCR text detection results for an image"""
    try:
        response = supabase.table(TEXT_ELEMENTS_TABLE).select('*').eq('image_id', image_id).execute()
        elements = []
        for record in response.data:
            # Convert database format back to analysis format
            element = {
                'text': record['text_content'],
                'confidence': float(record['confidence']),
                'bbox': [
                    float(record['bbox_x']),
                    float(record['bbox_y']),
                    float(record['bbox_x'] + record['bbox_width']),
                    float(record['bbox_y'] + record['bbox_height'])
                ]
            }
            elements.append(element)
        return elements
    except Exception as e:
        logger.error(f"Error fetching text elements for image {image_id}: {e}")
        return []

def fetch_preprocessed_science_results(image_id: str):
    """Fetch all preprocessed science analysis results for an image"""
    try:
        # Fetch chart analysis
        chart_analysis = fetch_chart_analysis(image_id)
        
        # Fetch chart elements (includes both chart elements and data points)
        chart_elements = fetch_chart_elements(image_id)
        
        # Fetch text elements
        text_elements = fetch_text_elements(image_id)
        
        # Combine results
        results = {
            'chart_type': chart_analysis['chart_type'] if chart_analysis else None,
            'chart_type_confidence': float(chart_analysis['confidence']) if chart_analysis else None,
            'chart_elements': chart_elements,
            'text_elements': text_elements
        }
        
        return results
    except Exception as e:
        logger.error(f"Error fetching preprocessed science results for image {image_id}: {e}")
        return None

# ─── HELPERS ───────────────────────────────────────────────────────────────
def fetch_scientific_tasks():
    """Fetch tasks where isReady is false and category is Scientific Figures"""
    # fetch category ID for "Scientific Figures"
    resp = supabase.table(CATEGORIES_TABLE).select('id').eq('name', 'Scientific Figures').execute()
    if not resp.data:
        logger.info("No category 'Scientific Figures' found.")
        return []
    cat_id = resp.data[0]['id']

    # fetch tasks where isReady is false and category matches
    resp = supabase.table(TASKS_TABLE).select('id').eq('category_id', cat_id).eq('isReady', False).execute()
    return [t['id'] for t in resp.data] if resp.data else []

def fetch_images_for_task(task_id):
    resp = supabase.table(IMAGES_TABLE).select('id, storage_link').eq('task_id', task_id).execute()
    return resp.data

def download_image(storage_link: str, bucket: str = IMAGE_BUCKET, expires_sec: int = 3600) -> np.ndarray:
    """
    Given a full public URL (storage_link), extract the internal path,
    create a signed URL (valid for expires_sec), fetch it, and decode to RGB.
    """
    # 1) parse out the bucket‐internal path
    p = urlparse(storage_link).path
    # strip '/storage/v1/object/public/'
    prefix = "/storage/v1/object/public/images/"
    assert prefix in p, f"unexpected storage_link: {storage_link}"
    internal_path = p.split(prefix, 1)[1]

    # 2) get signed URL
    logger.info(f"Getting signed URL for: {bucket}/{internal_path}")
    signed = supabase.storage.from_(bucket).create_signed_url(internal_path, expires_sec)
    signed_url = signed["signedUrl"]

    # 3) fetch bytes
    r = requests.get(signed_url)
    r.raise_for_status()
    buf = np.frombuffer(r.content, dtype=np.uint8)

    # 4) decode and convert to RGB
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to decode image from {storage_link}. Image may be corrupted or empty.")
    
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def upload_mask(image_id: str, mask: np.ndarray) -> str:
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    path = f"{image_id}/{uuid.uuid4().hex}.png"
    supabase.storage.from_(MASK_BUCKET).upload(path, buf.read())
    return path

def save_masks(image_id: str, masks: list):
    """Save SAM-generated masks"""
    logger.info(f"Saving {len(masks)} masks for image: {image_id}")
    records = []
    for m in masks:
        # Upload binary mask image to storage
        path = upload_mask(image_id, m['segmentation'])

        # Ensure all values are JSON-serializable (no numpy types)
        # area: scalar
        area = int(m.get('area', 0))

        # bbox: list/array of 4 integers (pixel coordinates)
        bbox_raw = m.get('bbox', [])
        if hasattr(bbox_raw, 'tolist'):
            bbox = [int(round(float(v))) for v in bbox_raw.tolist()]
        else:
            bbox = [int(round(float(v))) for v in bbox_raw] if bbox_raw else []

        # predicted_iou: scalar
        predicted_iou_raw = m.get('predicted_iou', 0.0)
        predicted_iou = float(predicted_iou_raw.item()) if hasattr(predicted_iou_raw, 'item') else float(predicted_iou_raw)

        # point_coords: list/array of shape (N, 2) or similar (pixel coordinates)
        point_coords_raw = m.get('point_coords')
        if point_coords_raw is None:
            point_coords = None
        elif hasattr(point_coords_raw, 'tolist'):
            # Convert nested lists to integers if they represent pixel coordinates
            coords_list = point_coords_raw.tolist()
            if isinstance(coords_list, list) and len(coords_list) > 0:
                if isinstance(coords_list[0], (list, tuple)):
                    point_coords = [[int(round(float(v))) for v in coord] for coord in coords_list]
                else:
                    point_coords = [int(round(float(v))) for v in coords_list]
            else:
                point_coords = coords_list
        else:
            # Convert to integers if it's a list/tuple
            if isinstance(point_coords_raw, (list, tuple)) and len(point_coords_raw) > 0:
                if isinstance(point_coords_raw[0], (list, tuple)):
                    point_coords = [[int(round(float(v))) for v in coord] for coord in point_coords_raw]
                else:
                    point_coords = [int(round(float(v))) for v in point_coords_raw]
            else:
                point_coords = point_coords_raw

        # stability_score: scalar
        stability_raw = m.get('stability_score', 0.0)
        stability_score = float(stability_raw.item()) if hasattr(stability_raw, 'item') else float(stability_raw)

        # crop_box: list/array of 4 integers (pixel coordinates)
        crop_box_raw = m.get('crop_box', [])
        if hasattr(crop_box_raw, 'tolist'):
            crop_box = [int(round(float(v))) for v in crop_box_raw.tolist()]
        else:
            crop_box = [int(round(float(v))) for v in crop_box_raw] if crop_box_raw else []

        record = {
            'image_id': image_id,
            'mask_url': path,
            'area': area,
            'bbox': bbox,
            'predicted_iou': predicted_iou,
            'point_coords': point_coords,
            'stability_score': stability_score,
            'crop_box': crop_box,
        }

        # Per-mask debug log to help trace storage + DB rows
        logger.info(
            "Prepared mask record for image %s: url=%s, area=%d, bbox=%s, "
            "pred_iou=%.4f, stability=%.4f, has_points=%s, crop_box=%s",
            image_id,
            record["mask_url"],
            record["area"],
            record["bbox"],
            record["predicted_iou"],
            record["stability_score"],
            record["point_coords"] is not None,
            record["crop_box"],
        )

        records.append(record)

    if records:
        try:
            supabase.table(MASKS_TABLE).insert(records, returning='minimal').execute()
            logger.info(f"Inserted {len(records)} mask records into {MASKS_TABLE}")
        except Exception as e:
            logger.error(f"Failed to insert mask records into {MASKS_TABLE}: {e}")
            import traceback
            logger.error(traceback.format_exc())


def upload_embedding(image_id: str, arr: np.ndarray) -> str:
    """
    Upload an embedding array to Supabase storage (embeddings bucket)
    and return the storage path.
    """
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32))
    buf.seek(0)
    path = f"{image_id}/{uuid.uuid4().hex}.npy"
    supabase.storage.from_(EMBED_BUCKET).upload(path, buf.read())
    return path

def save_chart_analysis(image_id: str, chart_type: str, confidence: float):
    """Save chart type classification results"""
    supabase.table(CHART_ANALYSIS_TABLE).insert({
        'image_id': image_id,
        'chart_type': chart_type,
        'confidence': confidence
    }).execute()

def save_chart_elements(image_id: str, elements: list):
    """Save chart element detection results"""
    if not elements:
        return
    
    records = []
    for element in elements:
        bbox = element.get('bbox', [])
        if len(bbox) == 4:
            # Convert numpy types to Python types for JSON serialization
            bbox_x = float(bbox[0]) if hasattr(bbox[0], 'item') else float(bbox[0])
            bbox_y = float(bbox[1]) if hasattr(bbox[1], 'item') else float(bbox[1])
            bbox_x2 = float(bbox[2]) if hasattr(bbox[2], 'item') else float(bbox[2])
            bbox_y2 = float(bbox[3]) if hasattr(bbox[3], 'item') else float(bbox[3])
            
            records.append({
                'image_id': image_id,
                'element_type': element.get('element_type', ''),
                'confidence': float(element.get('confidence', 0.0)),
                'bbox_x': bbox_x,
                'bbox_y': bbox_y,
                'bbox_width': bbox_x2 - bbox_x,
                'bbox_height': bbox_y2 - bbox_y
            })
    
    if records:
        supabase.table(CHART_ELEMENTS_TABLE).insert(records).execute()



def save_text_elements(image_id: str, text_elements: list):
    """Save OCR text detection results"""
    if not text_elements:
        return
    
    records = []
    for element in text_elements:
        bbox = element.get('bbox', [])
        if len(bbox) == 4:
            # Convert numpy types to Python types for JSON serialization
            bbox_x = float(bbox[0]) if hasattr(bbox[0], 'item') else float(bbox[0])
            bbox_y = float(bbox[1]) if hasattr(bbox[1], 'item') else float(bbox[1])
            bbox_x2 = float(bbox[2]) if hasattr(bbox[2], 'item') else float(bbox[2])
            bbox_y2 = float(bbox[3]) if hasattr(bbox[3], 'item') else float(bbox[3])
            
            records.append({
                'image_id': image_id,
                'text_content': element.get('text', ''),
                'confidence': float(element.get('confidence', 0.0)),
                'bbox_x': bbox_x,
                'bbox_y': bbox_y,
                'bbox_width': bbox_x2 - bbox_x,
                'bbox_height': bbox_y2 - bbox_y
            })
    
    if records:
        supabase.table(TEXT_ELEMENTS_TABLE).insert(records).execute()

def classify_chart_type_and_decide_pipeline(image_path: str, image_id: str):
    """
    Classify chart type using HF Space API and decide which pipeline to use:
    - Medical images -> MedSAM pipeline
    - Allowed scientific figures -> Science analyzer pipeline
    - Other scientific figures -> Skip science analyzer, only save chart type
    """
    try:
        logger.info(f"Calling HF Space API for chart type classification of image {image_id}")
        
        # Call HF Space API for chart type classification
        hf_response = call_hf_space_api(image_path)
        
        if not hf_response:
            logger.warning("HF Space API returned no response, using default SAM pipeline")
            return "sam_pipeline", None
        
        # Parse the HF Space response
        analysis_results = parse_hf_space_response(hf_response)
        
        if not analysis_results:
            logger.warning("Failed to parse HF Space response, using default SAM pipeline")
            return "sam_pipeline", None
        
        # Get chart type from analysis results
        chart_type = analysis_results.get('chart_type')
        chart_type_confidence = analysis_results.get('chart_type_confidence', 0.9)
        
        if not chart_type:
            logger.warning("No chart type detected, using default SAM pipeline")
            return "sam_pipeline", None
        
        logger.info(f"Chart type classification: {chart_type} (confidence: {chart_type_confidence:.3f})")
        
        # Save chart analysis results
        save_chart_analysis(image_id, chart_type, chart_type_confidence)
        
        # Decide pipeline based on chart type
        normalized_chart_type = chart_type.strip().lower()
        allowed_types = [
            'scatter plot',
            'line graph',
            'bar plot',
            'medical image',
            'histogram',
            'vector plot',
            'bubble chart'
        ]
        if "medical" in normalized_chart_type:
            logger.info("Medical image detected, using MedSAM pipeline")
            return "medsam_pipeline", chart_type
        elif normalized_chart_type in allowed_types:
            logger.info(f"Allowed scientific figure detected: {normalized_chart_type}, using science analyzer pipeline")
            return "science_analyzer_pipeline", chart_type
        else:
            logger.info(f"Non-allowed scientific figure detected: {normalized_chart_type}, skipping science analyzer pipeline.")
            return "skip_science_analyzer", chart_type
    except Exception as e:
        logger.error(f"Error in chart type classification: {e}")
        return "sam_pipeline", None

def process_with_sam_pipeline(image_id: str, img_np: np.ndarray, temp_image_path: str = None):
    """Process image with standard SAM pipeline (uses HuggingFace API or local fallback)"""
    try:
        # Generate masks using HuggingFace API (primary) or local model (fallback)
        masks = generate_masks(img_np, temp_image_path)
        
        if masks:
            logger.info(f"Generated {len(masks)} masks for image")
            save_masks(image_id, masks)
            logger.info("Masks saved successfully!")
        else:
            logger.warning("No masks generated - both HuggingFace API and local model failed")
    except Exception as e:
        logger.error(f"Error in SAM processing: {e}")

def process_with_medsam_pipeline(image_id: str, img_np: np.ndarray, temp_image_path: str = None):
    """Process image with MedSAM-like pipeline (masks + embeddings via HuggingFace API)"""
    try:
        # Generate SAM masks using HuggingFace API
        masks = generate_masks(img_np, temp_image_path)
        
        if masks:
            logger.info(f"Generated {len(masks)} masks for medical image")
            save_masks(image_id, masks)
        else:
            logger.warning("No masks generated for medical image")

        # Also generate and store an embedding for this medical/scientific image
        if temp_image_path:
            ok = generate_embedding_via_huggingface(temp_image_path, image_id)
            if ok:
                logger.info(f"Embedding generation completed successfully for {image_id}")
            else:
                logger.warning(f"Embedding generation failed for {image_id}")
    except Exception as e:
        logger.error(f"Error in MedSAM processing: {e}")

def process_with_science_analyzer_pipeline(image_id: str, img_np: np.ndarray, temp_image_path: str):
    """Process image with HF Space API pipeline"""
    try:
        logger.info(f"Calling HF Space API for image {image_id}")
        
        # Call HF Space API for full scientific analysis
        hf_response = call_hf_space_api(temp_image_path)
        
        if not hf_response:
            logger.error(f"HF Space API returned no response for image {image_id}")
            return
        
        # Parse the HF Space response
        analysis_results = parse_hf_space_response(hf_response)
        
        if not analysis_results:
            logger.error(f"Failed to parse HF Space response for image {image_id}")
            return
        
        # Save chart type if available
        chart_type = analysis_results.get('chart_type')
        if chart_type:
            save_chart_analysis(image_id, chart_type, analysis_results.get('chart_type_confidence', 0.9))
            logger.info(f"Saved chart type: {chart_type}")
        
        # Save chart elements
        chart_elements = analysis_results.get('chart_elements', [])
        if chart_elements:
            # Log confidence distribution
            confidences = [elem.get('confidence', 0) for elem in chart_elements]
            logger.info(f"Chart element confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
            logger.info(f"Chart elements with confidence >= 0.4: {sum(1 for c in confidences if c >= 0.4)}")
            logger.info(f"Chart elements with confidence >= 0.1: {sum(1 for c in confidences if c >= 0.1)}")
            
            save_chart_elements(image_id, chart_elements)
            logger.info(f"Saved {len(chart_elements)} chart elements")
        
        # Step 2: Use local EasyOCR for text detection
        logger.info(f"Running local EasyOCR for text detection on image {image_id}")
        text_elements = perform_local_ocr(img_np)
        
        if text_elements:
            # Log confidence distribution
            confidences = [elem.get('confidence', 0) for elem in text_elements]
            logger.info(f"Local OCR text element confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
            logger.info(f"Local OCR text elements with confidence >= 0.4: {sum(1 for c in confidences if c >= 0.4)}")
            logger.info(f"Local OCR text elements with confidence >= 0.1: {sum(1 for c in confidences if c >= 0.1)}")
            
            save_text_elements(image_id, text_elements)
            logger.info(f"Saved {len(text_elements)} text elements from local OCR")
        else:
            logger.info("No text elements detected by local OCR")
        
        logger.info("HF Space API + Local OCR pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in HF Space API processing: {e}")

def save_temp_image(img_np: np.ndarray, image_id: str) -> str:
    """Save image temporarily for analysis"""
    temp_path = f"temp_{image_id}.jpg"
    img_pil = Image.fromarray(img_np)
    img_pil.save(temp_path)
    return temp_path

def cleanup_temp_image(temp_path: str):
    """Remove temporary image file"""
    try:
        os.remove(temp_path)
    except:
        pass

# ─── PROCESSING ────────────────────────────────────────────────────────────
def process_task(task_id: str):
    """Enhanced preprocessing for a single task"""
    images = fetch_images_for_task(task_id)
    logger.info(f"Processing task {task_id} with {len(images)} images")
    
    for img_rec in images:
        img_id = img_rec['id']
        logger.info(f"Processing image {img_id}")
        temp_image_path = None
        
        try:
            # Download image
            img_np = download_image(img_rec['storage_link'])
            logger.info(f"Image {img_id} downloaded successfully!")
            
            # Save temporary image for analysis
            temp_image_path = save_temp_image(img_np, img_id)
            
            # Classify chart type and decide pipeline
            pipeline_type, chart_type = classify_chart_type_and_decide_pipeline(temp_image_path, img_id)
            
            # Execute appropriate pipeline
            if pipeline_type == "medsam_pipeline":
                process_with_medsam_pipeline(img_id, img_np, temp_image_path)
            elif pipeline_type == "science_analyzer_pipeline":
                process_with_science_analyzer_pipeline(img_id, img_np, temp_image_path)
            elif pipeline_type == "skip_science_analyzer":
                # logger.info(f"Skipping science analyzer for image {img_id} (chart type: {chart_type})")
                logger.info(f"Running medsam pipeline for image {img_id} (chart type: {chart_type})")
                process_with_medsam_pipeline(img_id, img_np, temp_image_path)
                # Only chart type is saved, nothing else to do
            else:  # default to SAM pipeline
                process_with_sam_pipeline(img_id, img_np, temp_image_path)
            
            # Cleanup
            if temp_image_path:
                cleanup_temp_image(temp_image_path)
            logger.info(f"Image {img_id} processed successfully!")
            
        except Exception as e:
            logger.error(f"Error processing image {img_id}: {e}")
            if temp_image_path:
                cleanup_temp_image(temp_image_path)

    # Mark task as ready
    try:
        supabase.table(TASKS_TABLE).update({'isReady': True}).eq('id', task_id).execute()
        logger.info(f"Task {task_id} marked as ready")
    except Exception as e:
        logger.error(f"Error marking task {task_id} as ready: {e}")

def process_task_async(task_id: str):
    """Process a task asynchronously (for background processing)"""
    try:
        logger.info(f"Starting async processing for task {task_id}")
        process_task(task_id)
        logger.info(f"Async processing completed for task {task_id}")
    except Exception as e:
        logger.error(f"Error in async processing for task {task_id}: {e}")

def main():
    """Main processing loop"""
    logger.info("Starting enhanced preprocessing pipeline...")
    
    tasks = fetch_scientific_tasks()
    logger.info(f"Found {len(tasks)} tasks to process")
    
    for task_id in tasks:
        logger.info(f"Processing task {task_id}")
        start_time = time.perf_counter()
        
        try:
            process_task(task_id)
            elapsed = time.perf_counter() - start_time
            logger.info(f"Completed task {task_id} in {elapsed:.2f}s")
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")

if __name__ == "__main__":
    main() 