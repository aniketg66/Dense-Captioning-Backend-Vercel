"""
HuggingFace Space Client for MedSAM Segmentation
Simplified version matching user's working sample exactly
"""
import json
import tempfile
import os
import numpy as np
from PIL import Image
import ssl

# Remove proxy env vars that cause issues
_proxy_vars_to_remove = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy']
for var in _proxy_vars_to_remove:
    if var in os.environ:
        del os.environ[var]

try:
    from gradio_client import Client, handle_file
    
    # Patch websockets to fix SSL certificate verification issues (local dev on macOS)
    try:
        import websockets.legacy.client as ws_client
        _original_connect = ws_client.connect
        
        def _patched_websocket_connect(*args, **kwargs):
            """Patch websocket connect to disable SSL verification for local dev"""
            if 'ssl' not in kwargs or kwargs.get('ssl') is None:
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                kwargs['ssl'] = ssl_context
            return _original_connect(*args, **kwargs)
        
        ws_client.connect = _patched_websocket_connect
        print("✅ [utils/huggingface_client] Patched websockets.legacy.client.connect to fix SSL verification")
    except (ImportError, AttributeError):
        try:
            import websockets
            _original_connect = websockets.connect
            
            def _patched_websocket_connect(*args, **kwargs):
                if 'ssl' not in kwargs or kwargs.get('ssl') is None:
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
                    kwargs['ssl'] = ssl_context
                return _original_connect(*args, **kwargs)
            
            websockets.connect = _patched_websocket_connect
            print("✅ [utils/huggingface_client] Patched websockets.connect to fix SSL verification")
        except ImportError:
            pass
    
    GRADIO_CLIENT_AVAILABLE = True
except ImportError as e:
    Client = None
    handle_file = None
    GRADIO_CLIENT_AVAILABLE = False
    print(f"⚠️  [utils/huggingface_client] gradio_client not available: {e}")


class HuggingFaceSegmentationClient:
    """Client for calling HuggingFace Space segmentation API - matches user's working sample"""
    
    def __init__(self, space_url="Aniketg6/medsam-inference"):
        """Initialize the HuggingFace client
        
        Args:
            space_url: The HuggingFace Space URL (e.g., "Aniketg6/medsam-inference")
        """
        self.space_url = space_url
        self.client = None
        self.current_image_path = None
        self.current_image = None
        
    def _ensure_client(self):
        """Lazily initialize the Gradio client - matches user's sample exactly"""
        if self.client is None:
            hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
            
            # Initialize exactly as shown in user's working sample
            if hf_token:
                self.client = Client(self.space_url)
            else:
                self.client = Client(self.space_url)
            
            print(f"✅ Connected to HuggingFace Space: {self.space_url}")
        return self.client
    
    def load_image(self, image_path, precomputed_embedding=None):
        """Load an image for segmentation"""
        try:
            img = Image.open(image_path)
            self.current_image = np.array(img.convert("RGB"))
            self.current_image_path = image_path
            return True
        except Exception as e:
            print(f"Failed to load image: {e}")
            return False
    
    def load_image_from_array(self, image_array):
        """Load an image from numpy array"""
        try:
            self.current_image = image_array
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                Image.fromarray(image_array).save(tmp.name)
                self.current_image_path = tmp.name
            return True
        except Exception as e:
            print(f"Failed to load image from array: {e}")
            return False
    
    def _get_image_path(self):
        """Get the current image path"""
        if self.current_image_path and os.path.exists(self.current_image_path):
            return self.current_image_path
        if self.current_image is not None:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                Image.fromarray(self.current_image).save(tmp.name)
                self.current_image_path = tmp.name
                return tmp.name
        raise ValueError("No image loaded")
    
    def segment_with_points(self, points, labels):
        """Segment image using point prompts - EXACTLY matches user's working sample"""
        try:
            client = self._ensure_client()
            image_path = self._get_image_path()
            
            # EXACT pattern from user's working sample
            result = client.predict(
                image=handle_file(image_path),
                request_json=json.dumps({
                    "points": points,
                    "labels": labels
                }),
                api_name="/segment_points"
            )
            
            # Parse response exactly as shown in user's sample
            data = json.loads(result)
            
            if data.get('success'):
                return {
                    'masks': data['masks'],
                    'confidences': data['confidences'],
                    'method': data.get('method', 'huggingface_points')
                }
            else:
                print(f"Segmentation failed: {data.get('error')}")
                return None
                
        except Exception as e:
            print(f"Error calling HuggingFace segment_points API: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def segment_with_box(self, bbox):
        """Segment image using a single bounding box"""
        try:
            client = self._ensure_client()
            image_path = self._get_image_path()
            
            # Normalize bbox format
            if isinstance(bbox, dict):
                bbox_list = [bbox.get('x1', 0), bbox.get('y1', 0), 
                            bbox.get('x2', 0), bbox.get('y2', 0)]
            else:
                bbox_list = list(bbox)
            
            result = client.predict(
                image=handle_file(image_path),
                request_json=json.dumps({"bbox": bbox_list}),
                api_name="/segment_box"
            )
            
            data = json.loads(result)
            
            if data.get('success'):
                return {
                    'mask': data['mask'],
                    'confidence': data['confidence'],
                    'method': data.get('method', 'huggingface_box')
                }
            else:
                print(f"Segmentation failed: {data.get('error')}")
                return None
                
        except Exception as e:
            print(f"Error calling HuggingFace segment_box API: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def segment_with_multiple_boxes(self, bboxes):
        """Segment image using multiple bounding boxes - EXACTLY matches user's working sample"""
        try:
            client = self._ensure_client()
            image_path = self._get_image_path()
            
            # Normalize bbox format
            normalized_bboxes = []
            for bbox in bboxes:
                if isinstance(bbox, dict):
                    normalized_bboxes.append([
                        bbox.get('x1', 0), bbox.get('y1', 0),
                        bbox.get('x2', 0), bbox.get('y2', 0)
                    ])
                else:
                    normalized_bboxes.append(list(bbox))
            
            # EXACT pattern from user's working sample for multiple boxes
            result = client.predict(
                image=handle_file(image_path),
                request_json=json.dumps({
                    "bboxes": normalized_bboxes
                }),
                api_name="/segment_multiple_boxes"
            )
            
            # Parse response exactly as shown in user's sample
            data = json.loads(result)
            
            if data.get('success'):
                return {
                    'masks': data['masks'],
                    'confidences': data['confidences'],
                    'method': data.get('method', 'huggingface_multiple_boxes')
                }
            else:
                print(f"Segmentation failed: {data.get('error')}")
                return None
                
        except Exception as e:
            print(f"Error calling HuggingFace segment_multiple_boxes API: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def segment_with_points_and_box(self, points, labels, bbox):
        """Segment using both points and bounding box"""
        # Use box segmentation when both are provided (box is typically more accurate)
        return self.segment_with_box(bbox)
    
    @property
    def embedding(self):
        """Compatibility property - HuggingFace handles embeddings server-side"""
        return self.current_image is not None
    
    @property
    def device(self):
        """Compatibility property - HuggingFace handles device server-side"""
        return "huggingface_cloud"


# Singleton instance
_hf_client = None

def get_huggingface_client(space_url="Aniketg6/medsam-inference"):
    """Get or create the HuggingFace client singleton
    
    Args:
        space_url: The HuggingFace Space URL
        
    Returns:
        HuggingFaceSegmentationClient instance
    """
    global _hf_client
    if _hf_client is None:
        _hf_client = HuggingFaceSegmentationClient(space_url)
    return _hf_client
