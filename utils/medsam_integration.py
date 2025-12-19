import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import os
import sys
from skimage import transform, io

class MedSAMIntegrator:
    def __init__(self):
        """Initialize MedSAM with the actual MedSAM model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.medsam_model = None
        self.current_image = None
        self.current_image_path = None
        self.embedding = None
        
        # Initialize MedSAM model
        self._load_medsam_model()
    
    def _load_medsam_model(self):
        """Load the MedSAM model"""
        try:
            from segment_anything import sam_model_registry
            import torch

            # Check if MedSAM checkpoint exists
            medsam_ckpt_path = "models/medsam_vit_b.pth"
            if not os.path.exists(medsam_ckpt_path):
                print(f"⚠ MedSAM checkpoint not found at {medsam_ckpt_path}")
                print("Please download the MedSAM model first")
                return

            # Always load checkpoint with map_location to handle CPU-only devices
            checkpoint = torch.load(medsam_ckpt_path, map_location='cpu')
            self.medsam_model = sam_model_registry["vit_b"](checkpoint=None)
            self.medsam_model.load_state_dict(checkpoint)
            self.medsam_model.to(self.device)
            self.medsam_model.eval()
            print("✓ MedSAM model loaded successfully")

        except ImportError as e:
            print(f"⚠ segment_anything not available: {e}")
            print("MedSAM functionality will be disabled")
        except Exception as e:
            print(f"⚠ MedSAM model not available: {e}")
            print("MedSAM functionality will be disabled")
    
    def load_image(self, image_path, precomputed_embedding=None):
        """Load image for MedSAM processing"""
        try:
            # Load image
            img_np = io.imread(image_path)
            if len(img_np.shape) == 2:
                img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
            else:
                img_3c = img_np
            self.current_image = img_3c
            self.current_image_path = image_path
            
            print(f"Loaded image shape: {img_3c.shape} (H, W, C)")
            print(f"Image dimensions: {img_3c.shape[1]}x{img_3c.shape[0]} (WxH)")
            
            # Set precomputed embedding if available
            if precomputed_embedding is not None:
                success = self.set_precomputed_embedding(precomputed_embedding)
                if not success:
                    print("Failed to load precomputed embedding, calculating on-the-fly...")
                    self.get_embeddings()
            else:
                # Calculate embedding
                self.get_embeddings()
            
            print(f"✓ Image loaded for MedSAM: {img_3c.shape}")
            return True
                
        except Exception as e:
            print(f"Error loading image for MedSAM: {e}")
            return False
    
    @torch.no_grad()
    def get_embeddings(self):
        """Calculate image embedding for MedSAM"""
        if self.current_image is None:
            return None
        
        print("Calculating MedSAM embedding...")
        
        # Resize image to 1024x1024
        img_1024 = transform.resize(
            self.current_image, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        
        # Normalize to [0, 1]
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )
        
        # Convert to tensor (3, H, W)
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        )
        
        # Get embedding
        self.embedding = self.medsam_model.image_encoder(img_1024_tensor)
        print("✓ Embedding calculated")
    
    def set_precomputed_embedding(self, embedding_array):
        """Set precomputed embedding from numpy array"""
        try:
            # Convert numpy array to tensor and move to device
            if isinstance(embedding_array, np.ndarray):
                embedding_tensor = torch.tensor(embedding_array).to(self.device)
                self.embedding = embedding_tensor
                print(f"✓ Precomputed embedding loaded: {embedding_tensor.shape}")
                return True
            else:
                print("Error: embedding_array must be a numpy array")
                return False
        except Exception as e:
            print(f"Error setting precomputed embedding: {e}")
            return False
    
    @torch.no_grad()
    def medsam_inference(self, box_1024, height, width):
        """MedSAM inference with bounding box"""
        if self.embedding is None or self.medsam_model is None:
            return None
        
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=self.embedding.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)
        
        sparse_embeddings, dense_embeddings = self.medsam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        
        low_res_logits, _ = self.medsam_model.mask_decoder(
            image_embeddings=self.embedding,  # (B, 256, 64, 64)
            image_pe=self.medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        
        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
        
        low_res_pred = F.interpolate(
            low_res_pred,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, gt.shape)
        
        low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (height, width)
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
        
        return medsam_seg
    
    def segment_with_box(self, bbox):
        """
        Segment image using bounding box
        bbox: [x1, y1, x2, y2] coordinates in original image space
        """
        if self.embedding is None or self.current_image is None:
            return None
        
        try:
            H, W, _ = self.current_image.shape
            print(f"Original image dimensions: {W}x{H}")
            print(f"Received bbox: {bbox}")
            
            # Ensure bbox coordinates are within image bounds
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, W-1))
            y1 = max(0, min(y1, H-1))
            x2 = max(0, min(x2, W-1))
            y2 = max(0, min(y2, H-1))
            
            # Ensure x2 > x1 and y2 > y1
            if x2 <= x1:
                x2 = min(x1 + 10, W-1)
            if y2 <= y1:
                y2 = min(y1 + 10, H-1)
            
            bbox = [x1, y1, x2, y2]
            print(f"Adjusted bbox: {bbox}")
            
            # Convert bbox to 1024 scale
            box_np = np.array([bbox])
            print(f"Original bbox array: {box_np}")
            print(f"Scaling factors: W={W}, H={H}")
            print(f"Scaling array: [{W}, {H}, {W}, {H}]")
            
            box_1024 = box_np / np.array([W, H, W, H]) * 1024
            print(f"Box in 1024 scale: {box_1024[0]}")
            
            # Check if coordinates are reasonable
            if np.any(box_1024 < 0) or np.any(box_1024 > 1024):
                print(f"WARNING: Box coordinates out of 1024 range: {box_1024[0]}")
            
            # Perform MedSAM inference
            medsam_mask = self.medsam_inference(box_1024, H, W)
            
            if medsam_mask is not None:
                return {
                    'mask': medsam_mask,
                    'confidence': 1.0,  # MedSAM doesn't provide confidence scores
                    'method': 'medsam_box'
                }
            else:
                return None
                
        except Exception as e:
            print(f"Error in MedSAM box-based segmentation: {e}")
            return None
    
    def segment_with_points(self, points, labels):
        """
        Segment image using point prompts (not implemented in original MedSAM GUI)
        For now, we'll use a bounding box around the points
        """
        if not points or self.embedding is None:
            return None
        
        try:
            H, W, _ = self.current_image.shape
            print(f"Original image dimensions: {W}x{H}")
            print(f"Received points: {points}")
            print(f"Received labels: {labels}")
            
            # Convert points to bounding box
            points_array = np.array(points)
            x_min, y_min = points_array.min(axis=0)
            x_max, y_max = points_array.max(axis=0)
            
            print(f"Points min/max: ({x_min}, {y_min}) to ({x_max}, {y_max})")
            
            # Add some padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(W-1, x_max + padding)
            y_max = min(H-1, y_max + padding)
            
            bbox = [x_min, y_min, x_max, y_max]
            print(f"Generated bbox from points: {bbox}")
            
            return self.segment_with_box(bbox)
            
        except Exception as e:
            print(f"Error in MedSAM point-based segmentation: {e}")
            return None
    
    def segment_with_points_and_box(self, points, labels, bbox):
        """
        Segment image using both points and bounding box
        For MedSAM, we'll prioritize the bounding box
        """
        return self.segment_with_box(bbox)
    
    def is_available(self):
        """Check if MedSAM is available"""
        return self.medsam_model is not None 
    
    def generate_automatic_masks(self, image_path=None):
        """Generate all masks for an image using SamAutomaticMaskGenerator"""
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            import cv2
            import time
            if image_path is None:
                image_path = self.current_image_path
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            tick1 = time.perf_counter()
            sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth")
            mask_generator = SamAutomaticMaskGenerator(sam)
            print(f"Sam model loaded in {time.perf_counter() - tick1} seconds")
            tick2 = time.perf_counter()
            masks = mask_generator.generate(image)
            print(f"Generated automatic {len(masks)} masks in {time.perf_counter() - tick2} seconds")
            return masks
        except Exception as e:
            print(f"Error in automatic mask generation: {e}")
            return [] 