"""
End-to-end test script for Dense-Captioning-Toolkit backend + HuggingFace Space.

Workflow:
1. Create a dummy task and image in Supabase using `sampleimage.jpeg`
2. Call backend `/api/trigger-preprocessing/<task_id>` to run enhanced_preprocessing
   (this will call the HF Space to generate auto masks + embeddings)
3. Poll Supabase until the task's `isReady` flag becomes True
4. Call `/api/medsam/load_from_supabase` to load the image and warm up the HF client
5. Call `/api/medsam/segment_points` with a few points, overlay the masks, save `points_result.png`
6. Call `/api/medsam/segment_multiple_boxes` with a few boxes, overlay the masks, save `boxes_result.png`

Assumptions:
- Backend `app.py` is running locally at http://127.0.0.1:5000
- Supabase env vars are set (REACT_APP_SUPABASE_URL / REACT_APP_SUPABASE_ANON_KEY)
- The Supabase schema matches the project's `enhanced_preprocessing.py` expectations
- `sampleimage.jpeg` exists in the `backend/` directory
"""

import base64
import io
import json
import time
import uuid
from pathlib import Path
from typing import List, Tuple

import numpy as np
import requests
from PIL import Image
from supabase import create_client, Client
from urllib.parse import urlparse
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND_BASE_URL = "http://127.0.0.1:5000"
SAMPLE_IMAGE_PATH = Path("sampleimage.jpeg")

TASKS_TABLE = "tasks"
IMAGES_TABLE = "images"
CATEGORIES_TABLE = "categories"
IMAGE_BUCKET = "images"
MASK_BUCKET = "masks"
EMBED_BUCKET = "embeddings"

SCIENTIFIC_CATEGORY_NAME = "Scientific Figures"


def get_supabase_client() -> Client:
    """Create a Supabase client using the same env vars as the backend."""
    import os

    url = os.getenv("REACT_APP_SUPABASE_URL") or os.getenv("SUPABASE_URL")
    key = os.getenv("REACT_APP_SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError(
            "Supabase URL/KEY not set. Please export REACT_APP_SUPABASE_URL and "
            "REACT_APP_SUPABASE_ANON_KEY (or SUPABASE_URL / SUPABASE_KEY)."
        )
    print(f"Using Supabase URL: {url}")
    return create_client(url, key)


def get_scientific_category_id(supabase: Client) -> str:
    """Fetch the category_id for 'Scientific Figures'."""
    print(f"Fetching category id for '{SCIENTIFIC_CATEGORY_NAME}'...")
    resp = supabase.table(CATEGORIES_TABLE).select("id").eq(
        "name", SCIENTIFIC_CATEGORY_NAME
    ).execute()
    if not resp.data:
        raise RuntimeError(
            f"Category '{SCIENTIFIC_CATEGORY_NAME}' not found in {CATEGORIES_TABLE}"
        )
    cat_id = resp.data[0]["id"]
    print(f"✓ Found category id: {cat_id}")
    return cat_id


def upload_image_to_supabase(supabase: Client, image_path: Path, image_id: str) -> str:
    """
    Upload local image file to Supabase storage and return the public storage_link
    that matches what `enhanced_preprocessing.download_image` expects **and**
    complies with the current RLS policy on the `images` bucket:

    ((bucket_id = 'images')
      AND storage.extension(name) = 'jpg'
      AND lower((storage.foldername(name))[1]) = 'public'
      AND auth.role() = 'anon')

    So we must:
      - use a path starting with 'public/...'
      - ensure the object name ends with '.jpg'
    """
    # Always upload as JPEG into the `public/` folder so it matches your RLS policy
    # and the SupabaseManager path parsing logic (expects 'public/<image_id>.jpg').
    storage_key = f"public/{image_id}.jpg"
    print(f"Uploading image to Supabase storage: {storage_key}")

    # Convert the local file to JPEG bytes (even if the source is .jpeg/.png/etc.)
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    supabase.storage.from_(IMAGE_BUCKET).upload(storage_key, buf.read())

    # Build public storage_link as used by enhanced_preprocessing.download_image
    url = supabase.supabase_url if hasattr(supabase, "supabase_url") else supabase._client_params["supabase_url"]  # type: ignore[attr-defined]
    storage_link = f"{url}/storage/v1/object/public/{IMAGE_BUCKET}/{storage_key}"
    print(f"✓ Uploaded. storage_link: {storage_link}")
    return storage_link


def create_dummy_task_and_image(supabase: Client) -> Tuple[str, str]:
    """
    Create a dummy task and image in Supabase for sampleimage.jpeg.

    Returns:
        (task_id, image_id)
    """
    if not SAMPLE_IMAGE_PATH.exists():
        raise FileNotFoundError(f"{SAMPLE_IMAGE_PATH} does not exist")

    cat_id = get_scientific_category_id(supabase)

    task_id = str(uuid.uuid4())
    image_id = str(uuid.uuid4())

    print(f"Creating dummy task {task_id} and image {image_id}...")

    # Insert task
    # Your `tasks` table requires a non-null `name` column (see error
    # "null value in column \"name\" of relation \"tasks\""). We therefore
    # provide a simple test name, plus the required foreign key + isReady.
    from datetime import datetime, timezone

    # Use an immediate due_date (UTC ISO 8601 string) to satisfy NOT NULL constraint
    due_date = datetime.now(timezone.utc).isoformat()

    supabase.table(TASKS_TABLE).insert(
        {
            "id": task_id,
            "category_id": cat_id,
            "isReady": False,
            "name": f"Test Task {task_id}",
            "due_date": due_date,
        }
    ).execute()

    # Upload image and insert image row
    storage_link = upload_image_to_supabase(supabase, SAMPLE_IMAGE_PATH, image_id)
    # Insert image row. The backend only requires id, task_id and storage_link
    # (see enhanced_preprocessing.fetch_images_for_task), so we avoid any
    # optional columns that may not exist in your schema.
    supabase.table(IMAGES_TABLE).insert(
        {
            "id": image_id,
            "task_id": task_id,
            "storage_link": storage_link,
        }
    ).execute()

    print(f"✓ Created task and image in Supabase")
    return task_id, image_id


def trigger_preprocessing(task_id: str) -> None:
    """Call backend /api/trigger-preprocessing/<task_id>."""
    url = f"{BACKEND_BASE_URL}/api/trigger-preprocessing/{task_id}"
    print(f"Calling {url} ...")
    resp = requests.post(url)
    print(f"Trigger response: {resp.status_code} {resp.text}")
    resp.raise_for_status()


def wait_for_task_ready(supabase: Client, task_id: str, timeout_sec: int = 900) -> None:
    """Poll Supabase until task.isReady is True or timeout."""
    print(f"Waiting for task {task_id} to complete preprocessing...")
    start = time.time()
    while True:
        resp = supabase.table(TASKS_TABLE).select("isReady").eq("id", task_id).execute()
        if resp.data:
            is_ready = resp.data[0].get("isReady")
            print(f"  isReady={is_ready}")
            if is_ready:
                print("✓ Task preprocessing completed")
                return
        if time.time() - start > timeout_sec:
            raise TimeoutError(f"Timeout waiting for task {task_id} to become ready")
        time.sleep(10)


def backend_medsam_load_image(image_id: str) -> dict:
    """Call /api/medsam/load_from_supabase to load image & warm up HF client."""
    url = f"{BACKEND_BASE_URL}/api/medsam/load_from_supabase"
    print(f"Calling {url} for image_id={image_id} ...")
    resp = requests.post(url, json={"image_id": image_id})
    print(f"load_from_supabase status: {resp.status_code}")
    resp.raise_for_status()
    data = resp.json()
    print(f"load_from_supabase response keys: {list(data.keys())}")
    return data


def download_image_from_storage(supabase: Client, storage_link: str) -> Image.Image:
    """
    Download an image from Supabase. Handles both public storage links and signed URLs.
    Matches the frontend pattern from TaskView.js and EditTask.js.
    """
    print(f"Downloading image from storage_link: {storage_link}")
    
    # If it's already a signed URL (contains token parameter), use it directly
    if "?token=" in storage_link or "/sign/" in storage_link:
        print("  Using signed URL directly")
        signed_url = storage_link
    else:
        # Extract the internal path from public storage link
        # Format: https://.../storage/v1/object/public/images/{path}
        p = urlparse(storage_link).path
        prefix = f"/storage/v1/object/public/{IMAGE_BUCKET}/"
        if prefix not in p:
            raise ValueError(f"Unexpected storage_link format: {storage_link}")
        internal_path = p.split(prefix, 1)[1]
        
        # Generate a signed URL
        print(f"  Creating signed URL for path: {internal_path}")
        signed = supabase.storage.from_(IMAGE_BUCKET).create_signed_url(internal_path, 3600)
        signed_url = signed["signedUrl"]
    
    # Download the image
    r = requests.get(signed_url)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    print(f"✓ Downloaded image size: {img.size}")
    return img


def call_segment_points(points: List[Tuple[int, int]], labels: List[int]) -> dict:
    """Call /api/medsam/segment_points (MedSAM-based predictor)."""
    url = f"{BACKEND_BASE_URL}/api/medsam/segment_points"
    payload = {"points": points, "labels": labels}
    print(f"Calling {url} with points={points}, labels={labels}")
    resp = requests.post(url, json=payload)
    print(f"segment_points status: {resp.status_code}")
    resp.raise_for_status()
    data = resp.json()
    print(f"segment_points success={data.get('success')}, "
          f"num_masks={len(data.get('masks', []))}")
    return data


def call_segment_multiple_boxes(bboxes: List[List[int]]) -> dict:
    """Call /api/medsam/segment_multiple_boxes (MedSAM-based predictor)."""
    url = f"{BACKEND_BASE_URL}/api/medsam/segment_multiple_boxes"
    payload = {"bboxes": bboxes}
    print(f"Calling {url} with bboxes={bboxes}")
    resp = requests.post(url, json=payload)
    print(f"segment_multiple_boxes status: {resp.status_code}")
    resp.raise_for_status()
    data = resp.json()
    print(f"segment_multiple_boxes success={data.get('success')}, "
          f"num_masks={len(data.get('masks', []))}")
    return data


def overlay_masks_and_save(
    img: Image.Image,
    masks: List[dict],
    title: str,
    output_path: Path,
) -> None:
    """Overlay boolean masks on an image and save to disk."""
    img_np = np.array(img)
    H, W = img_np.shape[:2]

    plt.figure(figsize=(10, 8))
    plt.imshow(img_np)
    plt.axis("off")
    plt.title(title)

    for i, m in enumerate(masks):
        # For MedSAM endpoints, the key is 'mask'
        seg = m.get("segmentation")
        if seg is None:
            seg = m.get("mask")
        if seg is None:
            print(f"  [overlay] mask {i} has no 'mask' or 'segmentation' key, skipping")
            continue

        mask = np.array(seg, dtype=bool)
        if mask.shape[:2] != img_np.shape[:2]:
            print(
                f"  [overlay] mask {i} shape {mask.shape[:2]} "
                f"!= image shape {img_np.shape[:2]}, skipping"
            )
            continue

        color = np.random.rand(3)
        overlay = np.zeros((H, W, 4), dtype=np.uint8)
        overlay[mask] = [
            int(color[0] * 255),
            int(color[1] * 255),
            int(color[2] * 255),
            120,
        ]
        plt.imshow(overlay)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved visualization to {output_path}")


def main():
    supabase = get_supabase_client()

    # 1) Create dummy task + image in Supabase
    task_id, image_id = create_dummy_task_and_image(supabase)
    print(f"Task ID: {task_id}")
    print(f"Image ID: {image_id}")

    # 2) Trigger preprocessing via backend API
    trigger_preprocessing(task_id)

    # 3) Wait for preprocessing to complete (auto masks + embeddings via HF Space)
    wait_for_task_ready(supabase, task_id, timeout_sec=900)

    # 4) Call backend to load image from Supabase and warm up HF MedSAM predictor
    load_data = backend_medsam_load_image(image_id)
    storage_link = load_data.get("image_url") or load_data.get("image_data") or ""
    if not storage_link:
        # Fallback: try to read storage_link directly from Supabase images table
        print("No image_url/image_data in response; fetching storage_link from Supabase...")
        resp = supabase.table(IMAGES_TABLE).select("storage_link").eq("id", image_id).execute()
        if resp.data:
            storage_link = resp.data[0]["storage_link"]
        else:
            raise RuntimeError("Could not find storage_link for image in Supabase")

    # Download original image to overlay results
    img = download_image_from_storage(supabase, storage_link)

    # 5) Call segment_points (MedSAM) and save visualization
    #    Use a few example points near the center of the image
    w, h = img.size
    points = [
        (w // 4, h // 4),
        (w // 2, h // 2),
        (3 * w // 4, 3 * h // 4),
    ]
    labels = [1, 1, 1]

    seg_points = call_segment_points(points, labels)
    if not seg_points.get("success") or not seg_points.get("masks"):
        print("No masks from segment_points; skipping points visualization")
    else:
        overlay_masks_and_save(
            img,
            seg_points["masks"],
            title="MedSAM Segment Points",
            output_path=Path("points_result.png"),
        )

    # 6) Call segment_multiple_boxes (MedSAM) and save visualization
    bboxes = [
        [w // 8, h // 8, w // 2, h // 2],
        [w // 3, h // 3, w * 3 // 4, h * 3 // 4],
    ]
    seg_boxes = call_segment_multiple_boxes(bboxes)
    if not seg_boxes.get("success") or not seg_boxes.get("masks"):
        print("No masks from segment_multiple_boxes; skipping boxes visualization")
    else:
        overlay_masks_and_save(
            img,
            seg_boxes["masks"],
            title="MedSAM Segment Multiple Boxes",
            output_path=Path("boxes_result.png"),
        )

    print("\n=== DONE ===")
    print(f"Task ID:   {task_id}")
    print(f"Image ID:  {image_id}")
    print("Results:")
    print("  - points_result.png : MedSAM point-based segmentation")
    print("  - boxes_result.png  : MedSAM multi-box segmentation")


if __name__ == "__main__":
    main()


