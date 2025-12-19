# Backend Utils Package

This package contains utility modules for MedSAM integration.

## Modules

### `supabase_client.py`
Handles all Supabase database operations:
- Load images from storage
- Get/save/delete masks
- Load precomputed embeddings
- Parallel processing for large mask sets

### `medsam_integration.py`
Handles MedSAM model operations:
- Load and initialize MedSAM model
- Point-based segmentation
- Box-based segmentation
- Image embedding computation

### `config.py` (in parent directory)
Configuration management:
- Load environment variables from `.env`
- Provide Supabase credentials
- Define storage bucket names

## Usage

```python
from utils.supabase_client import SupabaseManager
from utils.medsam_integration import MedSAMIntegrator

# Initialize
supabase = SupabaseManager()
medsam = MedSAMIntegrator()

# Load image and masks
data = supabase.get_image_and_masks(image_id=123)

# Load image for segmentation
medsam.load_image(data['temp_image_path'], data['embedding'])

# Segment with box
result = medsam.segment_with_box([x1, y1, x2, y2])
mask = result['mask']
```

## Requirements

See `../requirements.txt` for all dependencies.

Key dependencies:
- `supabase-py` - Supabase client
- `python-dotenv` - Environment variable management
- `opencv-python` - Image processing
- `scikit-image` - Image transformations
- `torch` - PyTorch (optional, for MedSAM model)
- `segment-anything` - SAM/MedSAM models (optional)

