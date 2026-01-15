import os

# Supabase Configuration
# All sensitive values MUST come from environment variables (no hardcoded fallbacks)
# Support multiple env var names for flexibility

# Supabase URL - can have fallback to default (not sensitive)
SUPABASE_URL = (
    os.getenv("SUPABASE_URL") or 
    os.getenv("REACT_APP_SUPABASE_URL") or 
    "https://ayiwlxmvainywvpxxckn.supabase.co"
)

# Supabase Key - MUST come from environment (sensitive, no fallback)
SUPABASE_KEY = (
    os.getenv("SUPABASE_KEY") or 
    os.getenv("REACT_APP_SUPABASE_ANON_KEY") or 
    os.getenv("SUPABASE_ANON_KEY")
)

# Validate that we have valid values
if not SUPABASE_URL or not SUPABASE_URL.startswith("http"):
    raise ValueError(
        f"Invalid SUPABASE_URL: {SUPABASE_URL}. "
        f"Set SUPABASE_URL or REACT_APP_SUPABASE_URL environment variable."
    )

if not SUPABASE_KEY:
    raise ValueError(
        "SUPABASE_KEY is required but not set. "
        "Set one of these environment variables: "
        "SUPABASE_KEY, REACT_APP_SUPABASE_ANON_KEY, or SUPABASE_ANON_KEY"
    )

if len(SUPABASE_KEY) < 50:
    raise ValueError(
        f"SUPABASE_KEY appears to be invalid (too short: {len(SUPABASE_KEY)} chars). "
        "Check your environment variable."
    )

# Storage Buckets
STORAGE_BUCKET = "images"
MASKS_BUCKET = "masks"
EMBEDDINGS_BUCKET = "embeddings"

# Storage Paths
STORAGE_FOLDER = "medsam"
IMAGE_NAME = "matsci2.jpeg"

# Default Image ID (can be overridden)
IMAGE_ID = os.getenv("MEDSAM_IMAGE_ID", "4e7e7d67-9925-412c-a08d-10ceda1a0f81") 