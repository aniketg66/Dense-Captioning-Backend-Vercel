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
try:
    from gradio_client import Client as GradioClient, handle_file
    GRADIO_CLIENT_AVAILABLE = True
except ImportError:
    logger.warning("gradio_client not installed. Install with: pip install gradio_client")
    GRADIO_CLIENT_AVAILABLE = False

# ─── CONFIG ──────────────────────────────────────────────────────────────
SUPABASE_URL       = os.getenv("REACT_APP_SUPABASE_URL") or os.getenv("SUPABASE_URL")
SUPABASE_KEY       = os.getenv("REACT_APP_SUPABASE_ANON_KEY") or os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

# ─── LAZY CLIENT INIT ─────────────────────────────────────────────────────
# Clients are initialized lazily to avoid proxy/env-var issues at import time.
# On Render, proxy env vars can cause httpx (used by supabase/gotrue) to crash
# if they are still set when the client is created. app.py removes them at
# module level, but this file may be imported before that cleanup runs.

MEDSAM_HF_SPACE = os.getenv("MEDSAM_HF_SPACE", "Aniketg6/medsam-inference")
USE_HF_FOR_MASKS = os.getenv("USE_HF_FOR_MASKS", "true").lower() == "true"

_supabase_client = None
_hf_mask_client = None
_hf_mask_client_initialized = False


def _clean_proxy_env():
    """Remove proxy env vars that break httpx/gradio_client on Render"""
    for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
                'ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy']:
        os.environ.pop(var, None)


def _patch_httpx_proxy():
    """
    Monkey-patch httpx.Client and httpx.AsyncClient to accept (and ignore)
    the 'proxy' kwarg.  This fixes the version mismatch between gotrue
    (which passes proxy=) and older httpx versions (which don't accept it).
    """
    try:
        import httpx
        for cls in (httpx.Client, httpx.AsyncClient):
            original_init = cls.__init__
            if getattr(original_init, '_proxy_patched', False):
                continue

            def _make_patched(orig):
                def patched_init(self, *args, **kwargs):
                    kwargs.pop('proxy', None)
                    kwargs.pop('proxies', None)
                    return orig(self, *args, **kwargs)
                patched_init._proxy_patched = True
                return patched_init

            cls.__init__ = _make_patched(original_init)
        logger.info("Patched httpx Client/AsyncClient to accept proxy kwarg")
    except Exception as e:
        logger.warning(f"Could not patch httpx: {e}")


def get_supabase():
    """Lazily create Supabase client (after proxy env vars are cleaned)"""
    global _supabase_client
    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError(
                "Missing Supabase credentials. Set REACT_APP_SUPABASE_URL / "
                "REACT_APP_SUPABASE_ANON_KEY or SUPABASE_URL / SUPABASE_KEY."
            )
        _clean_proxy_env()
        _patch_httpx_proxy()
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized (lazy)")
    return _supabase_client


def get_hf_mask_client():
    """Lazily create HuggingFace mask client"""
    global _hf_mask_client, _hf_mask_client_initialized
    if _hf_mask_client_initialized:
        return _hf_mask_client
    _hf_mask_client_initialized = True

    if not GRADIO_CLIENT_AVAILABLE or not USE_HF_FOR_MASKS:
        return None

    try:
        _clean_proxy_env()
        logger.info(f"Connecting to HuggingFace Space: {MEDSAM_HF_SPACE}")
        _hf_mask_client = GradioClient(MEDSAM_HF_SPACE)
        try:
            status_result = _hf_mask_client.predict(api_name="/check_auto_mask_status")
            status = json.loads(status_result)
            if status.get('available'):
                logger.info(f"HuggingFace auto mask generation available (device: {status.get('device')})")
            else:
                logger.warning("HuggingFace Space connected but SAM-H model not loaded")
        except Exception as e:
            logger.warning(f"Could not check HuggingFace auto mask status: {e}")
    except Exception as e:
        logger.warning(f"Could not connect to HuggingFace Space: {e}")
        _hf_mask_client = None

    return _hf_mask_client

# ─── HF MASK GENERATION API ────────────────────────────────────────────────

def generate_masks_via_huggingface(image_path: str) -> list:
    """
    Generate automatic masks using HuggingFace Space API

    Args:
        image_path: Path to the image file

    Returns:
        List of masks in the same format as SamAutomaticMaskGenerator.generate()
    """
    client = get_hf_mask_client()
    if not client:
        logger.warning("HuggingFace mask client not available")
        return None

    try:
        logger.info(f"Calling HuggingFace API for automatic mask generation...")

        # Use the same parameter strategy as test_auto.py:
        # - max_masks = -1 => return all masks
        # - resize_longest = 512 => downscale long side for speed / smaller payload
        params = {"max_masks": -1, "resize_longest": 512}

        start = time.perf_counter()
        result = client.predict(
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
    client = get_hf_mask_client()
    if not client:
        logger.warning("HuggingFace mask client not available (cannot generate embeddings)")
        return False

    try:
        logger.info(f"Calling HuggingFace encode_image API for image_id={image_id}...")

        params = {"image_id": image_id}

        start = time.perf_counter()
        result = client.predict(
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
            get_supabase().table(EMBED_TABLE).insert(
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
    if get_hf_mask_client() and temp_image_path and USE_HF_FOR_MASKS:
        masks = generate_masks_via_huggingface(temp_image_path)
        if masks:
            logger.info("Using masks from HuggingFace API")
            return masks
        else:
            logger.error("HuggingFace API failed to generate masks.")
            return None

    logger.error("HuggingFace mask client not available or temp_image_path missing.")
    return None



# ─── FETCH FUNCTIONS ────────────────────────────────────────────────────────

def fetch_chart_analysis(image_id: str):
    """Fetch chart type classification results for an image"""
    try:
        response = get_supabase().table(CHART_ANALYSIS_TABLE).select('*').eq('image_id', image_id).execute()
        if response.data:
            return response.data[0]  # Return the first (and should be only) result
        return None
    except Exception as e:
        logger.error(f"Error fetching chart analysis for image {image_id}: {e}")
        return None

def fetch_chart_elements(image_id: str):
    """Fetch chart element detection results for an image (including data points)"""
    try:
        response = get_supabase().table(CHART_ELEMENTS_TABLE).select('*').eq('image_id', image_id).execute()
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
        response = get_supabase().table(TEXT_ELEMENTS_TABLE).select('*').eq('image_id', image_id).execute()
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
    resp = get_supabase().table(CATEGORIES_TABLE).select('id').eq('name', 'Scientific Figures').execute()
    if not resp.data:
        logger.info("No category 'Scientific Figures' found.")
        return []
    cat_id = resp.data[0]['id']

    # fetch tasks where isReady is false and category matches
    resp = get_supabase().table(TASKS_TABLE).select('id').eq('category_id', cat_id).eq('isReady', False).execute()
    return [t['id'] for t in resp.data] if resp.data else []

def fetch_images_for_task(task_id):
    resp = get_supabase().table(IMAGES_TABLE).select('id, storage_link').eq('task_id', task_id).execute()
    return resp.data

def download_image(storage_link: str, bucket: str = IMAGE_BUCKET, expires_sec: int = 3600) -> np.ndarray:
    """
    Given a full public URL (storage_link), extract the internal path,
    create a signed URL (valid for expires_sec), fetch it, and decode to RGB.
    """
    # 1) parse out the bucket‐internal path
    p = urlparse(storage_link).path
    # Try common path prefixes for Supabase storage URLs
    prefix = "/storage/v1/object/public/images/"
    if prefix not in p:
        # Try without the bucket name in case URL format differs
        alt_prefix = "/storage/v1/object/public/"
        if alt_prefix in p:
            # Extract everything after /public/ and strip the bucket name
            after_public = p.split(alt_prefix, 1)[1]
            # Remove bucket prefix if present (e.g., "images/task_id/file.jpg" -> "task_id/file.jpg")
            if after_public.startswith(f"{bucket}/"):
                internal_path = after_public[len(f"{bucket}/"):]
            else:
                internal_path = after_public
        else:
            raise ValueError(f"Unexpected storage_link format: {storage_link}")
    else:
        internal_path = p.split(prefix, 1)[1]

    # 2) get signed URL
    logger.info(f"Getting signed URL for: {bucket}/{internal_path}")
    signed = get_supabase().storage.from_(bucket).create_signed_url(internal_path, expires_sec)
    logger.info(f"Signed URL response keys: {list(signed.keys()) if isinstance(signed, dict) else type(signed)}")
    # Handle different supabase-py versions (signedUrl vs signedURL vs signed_url)
    signed_url = (
        signed.get("signedUrl")
        or signed.get("signedURL")
        or signed.get("signed_url")
    )
    if not signed_url:
        raise ValueError(f"Could not extract signed URL from response: {signed}")

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
    get_supabase().storage.from_(MASK_BUCKET).upload(path, buf.read())
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

        # Pre-generate a UUID so we can return it reliably
        mask_id = str(uuid.uuid4())

        record = {
            'id': mask_id,
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
            "Prepared mask record for image %s: id=%s, url=%s, area=%d, bbox=%s, "
            "pred_iou=%.4f, stability=%.4f, has_points=%s, crop_box=%s",
            image_id,
            mask_id,
            record["mask_url"],
            record["area"],
            record["bbox"],
            record["predicted_iou"],
            record["stability_score"],
            record["point_coords"] is not None,
            record["crop_box"],
        )

        records.append(record)

    saved_ids = [r['id'] for r in records]
    if records:
        try:
            get_supabase().table(MASKS_TABLE).insert(records).execute()
            logger.info(f"Inserted {len(records)} mask records into {MASKS_TABLE}, ids={saved_ids}")
        except Exception as e:
            logger.error(f"Failed to insert mask records into {MASKS_TABLE}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise  # Re-raise so caller knows the DB insert failed
    return saved_ids


def upload_embedding(image_id: str, arr: np.ndarray) -> str:
    """
    Upload an embedding array to Supabase storage (embeddings bucket)
    and return the storage path.
    """
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32))
    buf.seek(0)
    path = f"{image_id}/{uuid.uuid4().hex}.npy"
    get_supabase().storage.from_(EMBED_BUCKET).upload(path, buf.read())
    return path

def save_chart_analysis(image_id: str, chart_type: str, confidence: float):
    """Save chart type classification results"""
    get_supabase().table(CHART_ANALYSIS_TABLE).insert({
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
        get_supabase().table(CHART_ELEMENTS_TABLE).insert(records).execute()



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
        get_supabase().table(TEXT_ELEMENTS_TABLE).insert(records).execute()

def process_with_sam_pipeline(image_id: str, img_np: np.ndarray, temp_image_path: str = None) -> bool:
    """Generate masks via HuggingFace Space (Aniketg6/medsam-inference) and save them.
    Returns True if masks were saved."""
    try:
        masks = generate_masks(img_np, temp_image_path)

        if masks:
            logger.info(f"Generated {len(masks)} masks for image {image_id}")
            save_masks(image_id, masks)
            logger.info(f"Masks saved successfully for image {image_id}")
            return True
        else:
            logger.warning(f"No masks generated for image {image_id}")
            return False
    except Exception as e:
        logger.error(f"Error in mask generation for image {image_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

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
    """Enhanced preprocessing for a single task.
    Only marks the task as ready when ALL images have masks/elements saved."""
    images = fetch_images_for_task(task_id)
    logger.info(f"Processing task {task_id} with {len(images)} images")

    succeeded_count = 0
    failed_images = []

    for img_rec in images:
        img_id = img_rec['id']
        logger.info(f"Processing image {img_id}")
        temp_image_path = None

        try:
            # Download image
            img_np = download_image(img_rec['storage_link'])
            logger.info(f"Image {img_id} downloaded successfully!")

            # Save temporary image for HF API call
            temp_image_path = save_temp_image(img_np, img_id)

            # Generate masks via MedSAM HuggingFace Space (Aniketg6/medsam-inference)
            success = process_with_sam_pipeline(img_id, img_np, temp_image_path)

            if success:
                succeeded_count += 1
                logger.info(f"Image {img_id} processed successfully (masks saved)")
            else:
                failed_images.append(img_id)
                logger.warning(f"Image {img_id}: no masks were generated/saved")

            # Cleanup
            if temp_image_path:
                cleanup_temp_image(temp_image_path)

        except Exception as e:
            failed_images.append(img_id)
            logger.error(f"Error processing image {img_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if temp_image_path:
                cleanup_temp_image(temp_image_path)

    # Only mark task as ready if ALL images had masks/elements saved
    all_succeeded = len(images) > 0 and succeeded_count == len(images)
    if all_succeeded:
        try:
            get_supabase().table(TASKS_TABLE).update({'isReady': True}).eq('id', task_id).execute()
            logger.info(f"Task {task_id} marked as ready (all {len(images)} images processed successfully)")
        except Exception as e:
            logger.error(f"Error marking task {task_id} as ready: {e}")
    else:
        logger.error(
            f"Task {task_id} NOT marked as ready: {succeeded_count}/{len(images)} images succeeded. "
            f"Failed images: {failed_images}"
        )

def process_task_async(task_id: str):
    """Process a task asynchronously (for background processing)"""
    try:
        logger.info(f"Starting async processing for task {task_id}")
        process_task(task_id)
        logger.info(f"Async processing completed for task {task_id}")
    except Exception as e:
        logger.error(f"Error in async processing for task {task_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())

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