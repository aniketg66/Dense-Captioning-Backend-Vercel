import os
import io
import uuid
import time
import requests
import numpy as np
import cv2
import torch
from urllib.parse import urlparse
from skimage import transform
from PIL import Image
from supabase import create_client, Client
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ─── CONFIG ──────────────────────────────────────────────────────────────
SUPABASE_URL       = os.getenv("SUPABASE_URL")
SUPABASE_KEY       = os.getenv("SUPABASE_KEY")
MEDSAM_CHECKPOINT     = "models/medsam_vit_b.pth"     # path to sam_vit_h checkpoint
SAM_CHECKPOINT  = "models/sam_vit_h_4b8939.pth"   # path to medsam_vit_b checkpoint
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TASKS_TABLE        = "tasks"
CATEGORIES_TABLE   = "categories"
IMAGES_TABLE       = "images"
MASK_BUCKET        = "masks"
EMBED_BUCKET       = "embeddings"
MASKS_TABLE        = "masks2"
EMBED_TABLE        = "embeddings2"
IMAGE_BUCKET       = "images"

# ─── CLIENT INIT ───────────────────────────────────────────────────────────
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─── MODEL INIT ────────────────────────────────────────────────────────────
# SAM
sam_model = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
sam_model.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam_model)

# MedSAM
state_dict = torch.load(MEDSAM_CHECKPOINT, map_location=DEVICE)
medsam_model = sam_model_registry["vit_b"](checkpoint=None)
medsam_model.load_state_dict(state_dict)
medsam_model.to(DEVICE)
medsam_model.eval()

# ─── HELPERS ───────────────────────────────────────────────────────────────
def fetch_scientific_tasks():
    # fetch category ID for "Scientific Figures"
    resp = supabase.table(CATEGORIES_TABLE).select('id').eq('name', 'Scientific Figures').execute()
    if not resp.data:
        print("No category 'Scientific Figures' found.")
        return []
    cat_id = resp.data[0]['id']

    # fetch tasks where isReady is false and category matches
    resp = supabase.table(TASKS_TABLE).select('id').eq('category_id', cat_id).eq('isReady', False).execute()
    return [t['id'] for t in resp.data] if resp.data else []

def fetch_images_for_task(task_id):
    resp = supabase.table(IMAGES_TABLE).select('id, storage_link').eq('task_id', task_id).execute()
    return resp.data

def download_image(storage_link: str, bucket: str = IMAGE_BUCKET, expires_sec: int = 3600) -> np.ndarray:
    """
    Given a full public URL (storage_link), extract the internal path,
    create a signed URL (valid for expires_sec), fetch it, and decode to RGB.
    """
    # 1) parse out the bucket‐internal path
    #    e.g. storage_link 
    #     https://<project>.supabase.co/storage/v1/object/public/images/foo/bar.png
    #    internal path: images/foo/bar.png
    p = urlparse(storage_link).path
    # strip '/storage/v1/object/public/'
    prefix = "/storage/v1/object/public/images/"
    assert prefix in p, f"unexpected storage_link: {storage_link}"
    internal_path = p.split(prefix, 1)[1]

    # 2) get signed URL
    print(bucket, internal_path)
    signed = supabase.storage.from_(bucket).create_signed_url(internal_path, expires_sec)
    print(signed)
    signed_url = signed["signedUrl"]

    # 3) fetch bytes
    r = requests.get(signed_url)
    r.raise_for_status()
    buf = np.frombuffer(r.content, dtype=np.uint8)

    # 4) decode and convert to RGB
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def upload_mask(image_id: str, mask: np.ndarray) -> str:
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    path = f"{image_id}/{uuid.uuid4().hex}.png"
    supabase.storage.from_(MASK_BUCKET).upload(path, buf.read())
    return path

def save_masks(image_id: str, masks: list):
    print(f"Saving masks for image: {image_id}")
    records = []
    for m in masks:
        path = upload_mask(image_id, m['segmentation'])
        records.append({
            'image_id': image_id,
            'mask_url': path,
            'area': m['area'],
            'bbox': m['bbox'],
            'predicted_iou': m['predicted_iou'],
            'point_coords': m['point_coords'],
            'stability_score': m['stability_score'],
            'crop_box': m['crop_box']
        })
    supabase.table(MASKS_TABLE).insert(records, returning='minimal').execute()


def upload_embedding(image_id: str, arr: np.ndarray) -> str:
    buf = io.BytesIO(); np.save(buf, arr.astype(np.float32)); buf.seek(0)
    path = f"{image_id}/{uuid.uuid4().hex}.npy"
    supabase.storage.from_(EMBED_BUCKET).upload(path, buf.read())
    return path

def save_embedding(image_id: str, tensor):
    arr = tensor.squeeze(0).cpu().numpy()
    path = upload_embedding(image_id, arr)
    supabase.table(EMBED_TABLE).insert({'image_id': image_id, 'file_path': path}).execute()

# ─── PROCESSING ────────────────────────────────────────────────────────────
def process_task(task_id: str):
    images = fetch_images_for_task(task_id)
    for img_rec in images:
        img_id = img_rec['id']
        img_np = download_image(img_rec['storage_link'])
        print(f"Image {img_id} downloaded successfully!")

        # masks
        masks = mask_generator.generate(img_np)
        print("Mask Generated successfully!")
        save_masks(img_id, masks)
        print("Mask saved successfully!")

        # embeddings
        # preprocess
        img_resized = transform.resize(img_np, (1024,1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
        img_norm = (img_resized - img_resized.min()) / np.clip(img_resized.max()-img_resized.min(),1e-8,None)
        tensor = torch.tensor(img_norm).float().permute(2,0,1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = medsam_model.image_encoder(tensor)
        save_embedding(img_id, emb)
        print("Embeddings saved successfully!")

    # mark task ready
    supabase.table(TASKS_TABLE).update({'isReady': True}).eq('id', task_id).execute()

def main():
  tasks = fetch_scientific_tasks()
  for t in tasks:
      print(f"Processing task {t}")
      start = time.perf_counter()
      process_task(t)
      print(f"Completed {t} in {time.perf_counter()-start:.2f}s")

main()