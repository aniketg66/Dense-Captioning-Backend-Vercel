"""
MedSAM HuggingFace Space Client
Drop-in replacement for SAM predictor that calls your HF Space API

Usage in app.py:
    # Replace this:
    # from segment_anything import sam_model_registry, SamPredictor
    # sam = sam_model_registry["vit_b"](checkpoint="models/sam_vit_h_4b8939.pth")
    # sam.to(device=device)
    # sam_predictor = SamPredictor(sam)
    
    # With this:
    from medsam_space_client import MedSAMSpacePredictor
    sam_predictor = MedSAMSpacePredictor(
        "https://YOUR_USERNAME-medsam-inference.hf.space/api/predict"
    )
    
    # Everything else stays exactly the same!
    # sam_predictor.set_image(image_array)
    # masks, scores, _ = sam_predictor.predict(point_coords=..., point_labels=...)
"""

import requests
import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from typing import Tuple, Optional
import time


class MedSAMSpacePredictor:
    """
    Drop-in replacement for SAM predictor that uses HuggingFace Space API
    
    This class mimics the interface of segment_anything.SamPredictor
    so you can replace SAM calls without changing your code.
    """
    
    def __init__(self, space_url: str, max_retries: int = 3, retry_delay: int = 10):
        """
        Initialize the Space predictor
        
        Args:
            space_url: URL of your HuggingFace Space API endpoint
                      e.g., "https://username-medsam-inference.hf.space/api/predict"
            max_retries: Number of retries if Space is sleeping/loading
            retry_delay: Seconds to wait between retries
        """
        self.space_url = space_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.image_array = None
        
        print(f"✓ MedSAM Space Predictor initialized")
        print(f"  URL: {space_url}")
        print(f"  Max retries: {max_retries}")
        
    def set_image(self, image: np.ndarray) -> None:
        """
        Set the image for segmentation
        
        Args:
            image: numpy array of shape (H, W, 3) or (H, W) - RGB or grayscale image
        """
        self.image_array = image
        
    def predict(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        multimask_output: bool = True,
        return_logits: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Predict masks using HuggingFace Space API
        
        This matches the SAM predictor interface exactly!
        
        Args:
            point_coords: (N, 2) array of point coordinates in (x, y) format
            point_labels: (N,) array of point labels (1=foreground, 0=background)
            multimask_output: If True, returns 3 masks. If False, returns 1 mask.
            return_logits: Ignored (kept for API compatibility)
            
        Returns:
            masks: (num_masks, H, W) boolean array of predicted masks
            scores: (num_masks,) array of confidence scores
            logits: None (not supported via API)
        """
        if self.image_array is None:
            raise ValueError("Must call set_image() before predict()")
        
        for attempt in range(self.max_retries):
            try:
                # Convert numpy array to base64
                image = Image.fromarray(self.image_array)
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                # Prepare points JSON
                points_json = json.dumps({
                    "coords": point_coords.tolist(),
                    "labels": point_labels.tolist(),
                    "multimask_output": multimask_output
                })
                
                # Call Space API
                response = requests.post(
                    self.space_url,
                    json={
                        "data": [
                            f"data:image/png;base64,{img_base64}",
                            points_json
                        ]
                    },
                    timeout=120  # 2 minute timeout
                )
                    
                # Handle non-200 responses
                if response.status_code == 503:
                    # Space is sleeping/loading
                            if attempt < self.max_retries - 1:
                        print(f"⏳ Space is loading (attempt {attempt + 1}/{self.max_retries}), waiting {self.retry_delay}s...")
                                time.sleep(self.retry_delay)
                                continue
                    else:
                        raise Exception("Space is not available after multiple retries")
                
                if response.status_code != 200:
                    raise Exception(f"API returned status {response.status_code}: {response.text}")
                
                # Parse result
                result = response.json()
                
                # Gradio wraps output in data array
                if "data" not in result or len(result["data"]) == 0:
                    raise Exception(f"Unexpected API response format: {result}")
                
                output_json = result["data"][0]
                output = json.loads(output_json)
                
                if not output.get('success', False):
                    raise Exception(output.get('error', 'Unknown API error'))
                
                # Convert masks back to numpy arrays
                masks = []
                for mask_data in output['masks']:
                    mask = np.array(mask_data['mask_data'], dtype=bool)
                    masks.append(mask)
                
                masks = np.array(masks)
                scores = np.array(output['scores'])
                
                return masks, scores, None
                
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    print(f"⏳ Request timeout (attempt {attempt + 1}/{self.max_retries}), retrying...")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise Exception("MedSAM Space API timeout after multiple retries")
                        
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    print(f"⏳ Request failed (attempt {attempt + 1}/{self.max_retries}): {e}, retrying...")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise Exception(f"MedSAM Space API request failed: {str(e)}")
        
        # Should not reach here
        raise Exception("Failed to get prediction from MedSAM Space")


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    """
    Example of how to use MedSAMSpacePredictor
    """
    import sys
    
    # Configuration
    SPACE_URL = "https://YOUR_USERNAME-medsam-inference.hf.space/api/predict"
    
    if "YOUR_USERNAME" in SPACE_URL:
        print("❌ Please update SPACE_URL with your HuggingFace username!")
        sys.exit(1)
    
    # Example usage
    print("Example usage:")
    print()
    print("from medsam_space_client import MedSAMSpacePredictor")
    print("import numpy as np")
    print("from PIL import Image")
    print()
    print("# Initialize predictor")
    print(f'predictor = MedSAMSpacePredictor("{SPACE_URL}")')
    print()
    print("# Load image")
    print('image = np.array(Image.open("test.jpg"))')
    print()
    print("# Set image")
    print("predictor.set_image(image)")
    print()
    print("# Run prediction")
    print("masks, scores, _ = predictor.predict(")
    print("    point_coords=np.array([[200, 150]]),")
    print("    point_labels=np.array([1]),")
    print("    multimask_output=True")
    print(")")
    print()
    print("# Get best mask")
    print("best_mask = masks[np.argmax(scores)]")
    print("print(f'Best mask shape: {best_mask.shape}')")
    print("print(f'Best score: {scores.max():.4f}')")
