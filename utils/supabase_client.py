# IMPORTANT: Remove proxy environment variables BEFORE importing supabase
# Railway sets these, but httpx (used by supabase) doesn't support proxy parameter
# in the version being used, causing: TypeError: Client.__init__() got an unexpected keyword argument 'proxy'
import os

_proxy_vars_backup = {}
_proxy_vars_to_remove = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
for var in _proxy_vars_to_remove:
    if var in os.environ:
        _proxy_vars_backup[var] = os.environ.pop(var)

from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY, STORAGE_BUCKET, MASKS_BUCKET, EMBEDDINGS_BUCKET, STORAGE_FOLDER, IMAGE_NAME, IMAGE_ID
import tempfile
import requests
import numpy as np
from PIL import Image
import io
from typing import List, Dict, Any
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Restore proxy vars after imports (they're not needed for supabase)
for var, value in _proxy_vars_backup.items():
    os.environ[var] = value
_proxy_vars_backup = {}

class SupabaseManager:
    def __init__(self):
        """Initialize Supabase client"""
        # Remove proxy vars again before create_client() in case httpx reads them
        # when creating Client instances (not just at import time)
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
        saved_proxy_vars = {}
        
        try:
            # Save and remove proxy vars
            for var in proxy_vars:
                if var in os.environ:
                    saved_proxy_vars[var] = os.environ[var]
                    del os.environ[var]
            
            # Create Supabase client without proxy interference
            self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        finally:
            # Restore proxy vars
            for var, value in saved_proxy_vars.items():
                os.environ[var] = value
    
    def get_image_data(self, image_id: str) -> Dict[str, Any]:
        """Fetch image data from the images table"""
        try:
            response = self.supabase.table('images').select('*').eq('id', image_id).execute()
            if response.data:
                return response.data[0]
            else:
                raise ValueError(f"Image with id {image_id} not found")
        except Exception as e:
            print(f"Error fetching image data: {e}")
            raise
    
    def get_signed_url(self, storage_path: str, bucket: str = STORAGE_BUCKET, expires_in: int = 3600) -> str:
        """Generate a signed URL for accessing storage files"""
        try:
            response = self.supabase.storage.from_(bucket).create_signed_url(
                path=storage_path,
                expires_in=expires_in
            )
            return response['signedURL']
        except Exception as e:
            print(f"Error generating signed URL: {e}")
            raise
    
    def get_masks_for_image(self, image_id: str) -> List[Dict[str, Any]]:
        """Fetch all masks for a specific image from masks2 table"""
        try:
            response = self.supabase.table('masks2').select('*').eq('image_id', image_id).execute()
            return response.data
        except Exception as e:
            print(f"Error fetching masks: {e}")
            return []
    
    def get_embedding_for_image(self, image_id: str) -> Dict[str, Any]:
        """Fetch embedding data for a specific image from embeddings2 table"""
        try:
            response = self.supabase.table('embeddings2').select('*').eq('image_id', image_id).execute()
            if response.data:
                return response.data[0]
            else:
                print(f"No embedding found for image_id: {image_id}")
                return None
        except Exception as e:
            print(f"Error fetching embedding: {e}")
            return None
    
    def download_embedding_to_array(self, embedding_url: str) -> np.ndarray:
        """Download embedding file and convert to numpy array"""
        try:
            response = requests.get(embedding_url)
            response.raise_for_status()
            
            # Load numpy array from bytes
            embedding_array = np.load(io.BytesIO(response.content))
            return embedding_array
            
        except Exception as e:
            print(f"Error downloading embedding: {e}")
            raise
    
    def download_image_to_temp(self, signed_url: str) -> str:
        """Download image from signed URL to a temporary file"""
        try:
            response = requests.get(signed_url)
            response.raise_for_status()
            
            # Create temporary file
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"medsam_image_{os.getpid()}.png")
            
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            return temp_path
        except Exception as e:
            print(f"Error downloading image: {e}")
            raise
    
    def download_mask_to_array(self, mask_url: str) -> np.ndarray:
        """Download mask image and convert to boolean array"""
        try:
            response = requests.get(mask_url)
            response.raise_for_status()
            
            # Load image and convert to boolean array
            img = Image.open(io.BytesIO(response.content))
            img_array = np.array(img, dtype=np.uint8)
            
            # Convert to boolean mask (assuming white pixels are the mask)
            if len(img_array.shape) == 3:  # RGB image
                # Convert to grayscale and threshold
                gray = np.mean(img_array, axis=2).astype(np.uint8)
                mask = (gray > 127).astype(bool)
            else:  # Grayscale image
                mask = (img_array > 127).astype(bool)
            
            return mask
            
        except Exception as e:
            print(f"Error downloading mask: {e}")
            raise
    
    def process_single_mask(self, mask_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single mask (for parallel processing)"""
        try:
            # Generate fresh signed URL for the mask using the stored path
            mask_path = mask_data['mask_url']
            mask_signed_url = self.get_signed_url(mask_path, bucket=MASKS_BUCKET)
            
            # Download and convert mask to array
            mask_array = self.download_mask_to_array(mask_signed_url)
            
            # Ensure mask_array is a proper boolean array
            if not isinstance(mask_array, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(mask_array)}")
            
            return {
                **mask_data,  # Include all original data
                'mask': mask_array  # Keep as numpy array for now, convert to list later if needed
            }
        except Exception as e:
            print(f"Error processing mask {mask_data.get('id', 'unknown')}: {e}")
            return None
    
    def get_image_and_masks(self, image_id: str) -> Dict[str, Any]:
        """Get image data and all associated masks"""
        try:
            # Get image data
            image_data = self.get_image_data(image_id)
            print(image_data)
            # Generate signed URL for the image using the actual image path from database
            tick1 = time.perf_counter()
            
            # Extract storage path from storage_link URL
            storage_link = image_data.get('storage_link')
            if storage_link:
                # Parse the URL to extract the path
                # URL format: https://.../storage/v1/object/public/images/{task_id}/{filename}
                try:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(storage_link)
                    path_parts = parsed_url.path.split('/')
                    # Find the index after 'images' in the path
                    images_index = path_parts.index('images') if 'images' in path_parts else -1
                    if images_index != -1 and images_index + 2 < len(path_parts):
                        # Extract task_id and filename
                        task_id = path_parts[images_index + 1]
                        filename = path_parts[images_index + 2]
                        storage_path = f"{task_id}/{filename}"
                        print(f"Extracted storage path from URL: {storage_path}")
                    else:
                        raise ValueError(f"Could not parse storage path from URL: {storage_link}")
                except Exception as e:
                    print(f"Error parsing storage_link: {e}")
                    raise ValueError(f"Invalid storage_link format: {storage_link}")
            else:
                # Fallback to storage_path or file_path fields
                storage_path = image_data.get('storage_path', image_data.get('file_path'))
                if not storage_path:
                    raise ValueError(f"No storage path found in image data: {image_data}")
            
            print(f"Using image storage path: {storage_path}")
            signed_url = self.get_signed_url(storage_path)
            
            # Download image to temp file
            temp_image_path = self.download_image_to_temp(signed_url)
            print(f"Time to download image: {time.perf_counter() - tick1:.2f} seconds")
            
            # Get embedding data
            tick_emb = time.perf_counter()
            embedding_data = self.get_embedding_for_image(image_id)
            embedding_array = None
            if embedding_data:
                # Generate signed URL for the embedding file
                embedding_path = embedding_data['file_path']
                embedding_signed_url = self.get_signed_url(embedding_path, bucket=EMBEDDINGS_BUCKET)
                
                # Download and load embedding array
                embedding_array = self.download_embedding_to_array(embedding_signed_url)
                print(f"Time to download embedding: {time.perf_counter() - tick_emb:.2f} seconds")
            else:
                print("No precomputed embedding found, will calculate on-the-fly")
            
            # Get masks from masks2 table
            tick2 = time.perf_counter()
            masks_data = self.get_masks_for_image(image_id)
            print(f"Time to get masks data: {time.perf_counter() - tick2:.2f} seconds")
            
            # Download and convert each mask to array
            tick3 = time.perf_counter()
            masks_with_arrays = []
            
            # Use parallel processing for mask downloads
            max_workers = min(multiprocessing.cpu_count() * 2, len(masks_data), 10)  # Limit to reasonable number
            print(f"Processing {len(masks_data)} masks with {max_workers} parallel workers...")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all mask processing tasks
                future_to_mask = {
                    executor.submit(self.process_single_mask, mask_data): mask_data 
                    for mask_data in masks_data
                }
                
                # Collect results as they complete
                completed_count = 0
                for future in as_completed(future_to_mask):
                    completed_count += 1
                    mask_data = future_to_mask[future]
                    
                    try:
                        result = future.result()
                        if result is not None:
                            masks_with_arrays.append(result)
                        
                        if completed_count % 5 == 0 or completed_count == len(masks_data):
                            print(f"Processed {completed_count}/{len(masks_data)} masks...")
                            
                    except Exception as e:
                        print(f"Error processing mask {mask_data.get('id', 'unknown')}: {e}")
                        continue
            
            print(f"Time to process all masks: {time.perf_counter() - tick3:.2f} seconds")
            print(f"Successfully processed {len(masks_with_arrays)} out of {len(masks_data)} masks")
            
            return {
                'image_data': image_data,
                'signed_url': signed_url,
                'temp_image_path': temp_image_path,
                'masks': masks_with_arrays,
                'embedding': embedding_array
            }
        except Exception as e:
            print(f"Error getting image and masks: {e}")
            raise
    
    def convert_masks_to_lists(self, masks_with_arrays: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert numpy arrays to lists for JSON serialization"""
        print("Converting mask arrays to lists for JSON serialization...")
        tick = time.perf_counter()
        
        for mask_data in masks_with_arrays:
            if 'mask' in mask_data and hasattr(mask_data['mask'], 'tolist'):
                mask_data['mask'] = mask_data['mask'].tolist()
        
        print(f"Conversion completed in {time.perf_counter() - tick:.2f} seconds")
        return masks_with_arrays

    def get_image_and_basic_info(self, image_id: str) -> Dict[str, Any]:
        """Get image data and basic info without processing masks"""
        try:
            # Get image data
            image_data = self.get_image_data(image_id)
            print(image_data)
            
            # Generate signed URL for the image using the actual image path from database
            tick1 = time.perf_counter()
            
            # Extract storage path from storage_link URL
            storage_link = image_data.get('storage_link')
            if storage_link:
                # Parse the URL to extract the path
                # URL format: https://.../storage/v1/object/public/images/{task_id}/{filename}
                try:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(storage_link)
                    path_parts = parsed_url.path.split('/')
                    # Find the index after 'images' in the path
                    images_index = path_parts.index('images') if 'images' in path_parts else -1
                    if images_index != -1 and images_index + 2 < len(path_parts):
                        # Extract task_id and filename
                        task_id = path_parts[images_index + 1]
                        filename = path_parts[images_index + 2]
                        storage_path = f"{task_id}/{filename}"
                        print(f"Extracted storage path from URL: {storage_path}")
                    else:
                        raise ValueError(f"Could not parse storage path from URL: {storage_link}")
                except Exception as e:
                    print(f"Error parsing storage_link: {e}")
                    raise ValueError(f"Invalid storage_link format: {storage_link}")
            else:
                # Fallback to storage_path or file_path fields
                storage_path = image_data.get('storage_path', image_data.get('file_path'))
                if not storage_path:
                    raise ValueError(f"No storage path found in image data: {image_data}")
            
            print(f"Using image storage path: {storage_path}")
            signed_url = self.get_signed_url(storage_path)
            
            # Download image to temp file
            temp_image_path = self.download_image_to_temp(signed_url)
            print(f"Time to download image: {time.perf_counter() - tick1:.2f} seconds")
            
            # Get embedding data
            tick_emb = time.perf_counter()
            embedding_data = self.get_embedding_for_image(image_id)
            embedding_array = None
            if embedding_data:
                # Generate signed URL for the embedding file
                embedding_path = embedding_data['file_path']
                embedding_signed_url = self.get_signed_url(embedding_path, bucket=EMBEDDINGS_BUCKET)
                
                # Download and load embedding array
                embedding_array = self.download_embedding_to_array(embedding_signed_url)
                print(f"Time to download embedding: {time.perf_counter() - tick_emb:.2f} seconds")
            else:
                print("No precomputed embedding found, will calculate on-the-fly")
            
            return {
                'image_data': image_data,
                'signed_url': signed_url,
                'temp_image_path': temp_image_path,
                'embedding': embedding_array
            }
        except Exception as e:
            print(f"Error getting image and basic info: {e}")
            raise

    def get_mask_count(self, image_id: str) -> int:
        """Get the count of masks for an image without downloading them"""
        try:
            masks_data = self.get_masks_for_image(image_id)
            return len(masks_data)
        except Exception as e:
            print(f"Error getting mask count: {e}")
            return 0 