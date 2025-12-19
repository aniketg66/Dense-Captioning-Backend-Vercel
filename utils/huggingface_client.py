"""
HuggingFace Space Client for MedSAM Segmentation
Calls the HuggingFace Space API instead of using local models
"""
import json
import tempfile
import os
import numpy as np
from PIL import Image
from gradio_client import Client, handle_file


class HuggingFaceSegmentationClient:
    """Client for calling HuggingFace Space segmentation API"""
    
    def __init__(self, space_url="Aniketg6/medsam-inference"):
        """Initialize the HuggingFace client
        
        Args:
            space_url: The HuggingFace Space URL (e.g., "username/space-name")
        """
        self.space_url = space_url
        self.client = None
        self.current_image_path = None
        self.current_image = None
        self._initialized = False
        
    def _ensure_client(self):
        """Lazily initialize the Gradio client"""
        if self.client is None:
            print(f"Connecting to HuggingFace Space: {self.space_url}")
            try:
                self.client = Client(self.space_url)
                self._initialized = True
                print("✓ Connected to HuggingFace Space successfully!")
            except Exception as e:
                print(f"Failed to connect to HuggingFace Space: {e}")
                raise
        return self.client
    
    def is_available(self):
        """Check if the HuggingFace Space is available"""
        try:
            self._ensure_client()
            return True
        except:
            return False
    
    def load_image(self, image_path, precomputed_embedding=None):
        """Load an image for segmentation
        
        Args:
            image_path: Path to the image file
            precomputed_embedding: Not used (kept for API compatibility)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Verify image exists and can be opened
            img = Image.open(image_path)
            self.current_image = np.array(img.convert("RGB"))
            self.current_image_path = image_path
            print(f"Image loaded: {image_path}, shape: {self.current_image.shape}")
            return True
        except Exception as e:
            print(f"Failed to load image: {e}")
            return False
    
    def load_image_from_array(self, image_array):
        """Load an image from numpy array
        
        Args:
            image_array: Numpy array of the image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.current_image = image_array
            # Save to temporary file for API calls
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                Image.fromarray(image_array).save(tmp.name)
                self.current_image_path = tmp.name
            print(f"Image loaded from array, shape: {self.current_image.shape}")
            return True
        except Exception as e:
            print(f"Failed to load image from array: {e}")
            return False
    
    def _get_image_path(self):
        """Get the current image path, creating temp file if needed"""
        if self.current_image_path and os.path.exists(self.current_image_path):
            return self.current_image_path
        
        if self.current_image is not None:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                Image.fromarray(self.current_image).save(tmp.name)
                self.current_image_path = tmp.name
                return tmp.name
        
        raise ValueError("No image loaded")
    
    def segment_with_points(self, points, labels):
        """Segment image using point prompts
        
        Args:
            points: List of [x, y] coordinates
            labels: List of labels (1=foreground, 0=background)
            
        Returns:
            Dict with 'masks', 'confidences', 'method' or None on failure
        """
        try:
            client = self._ensure_client()
            image_path = self._get_image_path()
            
            request_json = json.dumps({
                "points": points,
                "labels": labels
            })
            
            print(f"Calling HuggingFace API: segment_points with {len(points)} points")
            
            result = client.predict(
                image=handle_file(image_path),
                request_json=request_json,
                api_name="/segment_points"
            )
            
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
        """Segment image using a single bounding box
        
        Args:
            bbox: [x1, y1, x2, y2] or {'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}
            
        Returns:
            Dict with 'mask', 'confidence', 'method' or None on failure
        """
        try:
            client = self._ensure_client()
            image_path = self._get_image_path()
            
            # Normalize bbox format
            if isinstance(bbox, dict):
                bbox_list = [bbox.get('x1', 0), bbox.get('y1', 0), 
                            bbox.get('x2', 0), bbox.get('y2', 0)]
            else:
                bbox_list = list(bbox)
            
            request_json = json.dumps({
                "bbox": bbox_list
            })
            
            print(f"Calling HuggingFace API: segment_box with bbox {bbox_list}")
            
            result = client.predict(
                image=handle_file(image_path),
                request_json=request_json,
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
        """Segment image using multiple bounding boxes
        
        Args:
            bboxes: List of [x1, y1, x2, y2] or [{'x1': ..., ...}, ...]
            
        Returns:
            Dict with 'masks', 'confidences', 'method' or None on failure
        """
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
            
            request_json = json.dumps({
                "bboxes": normalized_bboxes
            })
            
            print(f"Calling HuggingFace API: segment_multiple_boxes with {len(bboxes)} boxes")
            
            result = client.predict(
                image=handle_file(image_path),
                request_json=request_json,
                api_name="/segment_multiple_boxes"
            )
            
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
        """Segment using both points and bounding box
        
        Note: The HuggingFace Space doesn't have a combined endpoint,
        so we use the box segmentation as it's typically more accurate
        when a box is provided.
        
        Args:
            points: List of [x, y] coordinates (ignored in this implementation)
            labels: List of labels (ignored in this implementation)
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            Dict with 'mask', 'confidence', 'method' or None on failure
        """
        # Use box segmentation when both are provided
        # The box typically gives better results than points alone
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

