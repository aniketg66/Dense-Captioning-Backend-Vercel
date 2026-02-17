import os

# Supabase Configuration
SUPABASE_URL = os.getenv("REACT_APP_SUPABASE_URL") or os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("REACT_APP_SUPABASE_ANON_KEY") or os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")

# Storage Buckets
STORAGE_BUCKET = "images"
MASKS_BUCKET = "masks"
EMBEDDINGS_BUCKET = "embeddings"

# Storage Paths
STORAGE_FOLDER = "medsam"
IMAGE_NAME = "matsci2.jpeg"

# Default Image ID (can be overridden)
IMAGE_ID = os.getenv("MEDSAM_IMAGE_ID", "4e7e7d67-9925-412c-a08d-10ceda1a0f81") 