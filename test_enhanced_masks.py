import json
import time
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from enhanced_preprocessing import generate_masks_via_huggingface


def main():
    """
    Quick manual test for `generate_masks_via_huggingface` using a local image.

    Usage:
        cd backend
        python3 test_enhanced_masks.py sampleimage.jpeg
    """
    if len(sys.argv) < 2:
        print("Usage: python3 test_enhanced_masks.py /path/to/image.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    print(f"Testing generate_masks_via_huggingface on: {image_path}")

    start = time.perf_counter()
    masks = generate_masks_via_huggingface(str(image_path))
    elapsed = time.perf_counter() - start

    if masks is None:
        print("No masks returned (masks is None). Check logs for HF API errors.")
        return

    print(f"\nSuccess:   True")
    print(f"Num masks: {len(masks)}")
    print(f"API time:  {elapsed:.3f} seconds")

    if not masks:
        print("No masks generated.")
        return

    # Compute simple stats
    areas = [m["area"] for m in masks]
    ious = [m["predicted_iou"] for m in masks]
    stabilities = [m["stability_score"] for m in masks]
    print("\nMask Statistics:")
    print(f"  Areas:           {min(areas)} - {max(areas)} pixels")
    print(f"  IoU scores:      {min(ious):.3f} - {max(ious):.3f}")
    print(f"  Stability scores:{min(stabilities):.3f} - {max(stabilities):.3f}")

    # Load image and overlay masks; note generate_masks_via_huggingface uses resize_longest=512
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    # Downsize locally in the same way for visualization (longest side 512)
    h_local, w_local = img_np.shape[:2]
    longest = max(h_local, w_local)
    target_longest = 512
    if longest != target_longest:
        scale = target_longest / float(longest)
        new_w = max(1, int(w_local * scale))
        new_h = max(1, int(h_local * scale))
        print(f"\nResizing local image from {w_local}x{h_local} to {new_w}x{new_h} to match HF processing size...")
        img = img.resize((new_w, new_h))
        img_np = np.array(img)

    # Visualize
    plt.figure(figsize=(10, 10))
    plt.imshow(img_np)
    plt.axis("off")
    plt.title(f"Auto Masks via enhanced_preprocessing (N={len(masks)})")

    for i, m in enumerate(masks):
        seg = np.array(m["segmentation"], dtype=bool)
        if seg.shape[:2] != img_np.shape[:2]:
            print(f"Skipping mask {i}: seg shape {seg.shape[:2]} != image shape {img_np.shape[:2]}")
            continue

        color = np.random.rand(3)
        overlay = np.zeros((seg.shape[0], seg.shape[1], 4), dtype=np.uint8)
        overlay[seg] = [
            int(color[0] * 255),
            int(color[1] * 255),
            int(color[2] * 255),
            120,
        ]
        plt.imshow(overlay)

    plt.tight_layout()
    out_path = Path("enhanced_masks_test.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Saved visualization to: {out_path.resolve()}")


if __name__ == "__main__":
    main()


