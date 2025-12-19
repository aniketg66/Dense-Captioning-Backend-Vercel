import cv2
import numpy as np
import os

def extract_matplotlib_subfigures(
    image_path,
    output_folder="panels",
    bg_tol=30,
    gap_frac=0.02,
    select_frac=0.5,
    min_panel_frac=0.005
):
    """
    Extract R×C panels by selecting only the largest gutter gaps.

    Args:
      image_path:      path to your composite figure
      output_folder:   where to save panel_01.png, …
      bg_tol:          tolerance for matching bg color (0–255)
      gap_frac:        min gutter width as fraction of dim (for initial selection)
      select_frac:     fraction of max-gap to keep (e.g. 0.5 keeps only >50% of largest)
      min_panel_frac:  drop any panel whose area < this fraction of image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    H, W = img.shape[:2]

    # 1) Estimate bg color from border pixels
    top    = img[0:5,    :, :].reshape(-1,3)
    bottom = img[-5:,    :, :].reshape(-1,3)
    left   = img[:, 0:5, :].reshape(-1,3)
    right  = img[:, -5:, :].reshape(-1,3)
    border = np.vstack([top, bottom, left, right])
    bg_bgr = np.median(border, axis=0)
    bg_gray = int(np.dot(bg_bgr, [0.114,0.587,0.299]))

    # 2) background mask
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask_bg= np.abs(gray.astype(int) - bg_gray) <= bg_tol

    # 3) projection sums
    col_sum = mask_bg.sum(axis=0)  # length W
    row_sum = mask_bg.sum(axis=1)  # length H

    # thresholds for “mostly bg”
    col_thr = H * (1 - gap_frac)
    row_thr = W * (1 - gap_frac)

    # 4) find all gap segments with their widths
    def find_segments(proj, thr):
        segs = []
        in_gap, start = False, 0
        for i, v in enumerate(proj):
            if v >= thr and not in_gap:
                in_gap, start = True, i
            elif v < thr and in_gap:
                segs.append((start, i-1, i-1-start+1))
                in_gap = False
        if in_gap:
            segs.append((start, len(proj)-1, len(proj)-1-start+1))
        return segs

    v_segs = find_segments(col_sum, col_thr)
    h_segs = find_segments(row_sum, row_thr)

    if not v_segs or not h_segs:
        raise ValueError("No gutter segments found – try lowering gap_frac or increasing bg_tol")

    # 5) pick only the *largest* segments
    max_vw = max(w for _,_,w in v_segs)
    max_hw = max(w for _,_,w in h_segs)
    v_chosen = sorted([ (s,e,w) for s,e,w in v_segs if w >= max_vw*select_frac ], key=lambda x:x[0])
    h_chosen = sorted([ (s,e,w) for s,e,w in h_segs if w >= max_hw*select_frac ], key=lambda x:x[0])

    # if you get N chosen segments, you have N+1 panels in that direction
    v_centers = [ (s+e)//2 for s,e,_ in v_chosen ]
    h_centers = [ (s+e)//2 for s,e,_ in h_chosen ]

    xs = [0] + v_centers + [W]
    ys = [0] + h_centers + [H]

    # 6) slice panels, drop tiny
    panels = []
    total_area = H*W
    for yi in range(len(ys)-1):
        for xi in range(len(xs)-1):
            x0, x1 = xs[xi], xs[xi+1]
            y0, y1 = ys[yi], ys[yi+1]
            w, h = x1-x0, y1-y0
            if w*h >= total_area * min_panel_frac:
                panels.append((x0,y0,w,h))

    if not panels:
        raise RuntimeError("No panels left after min_panel_frac filtering")

    # 7) sort & save
    panels.sort(key=lambda b:(b[1],b[0]))
    os.makedirs(output_folder, exist_ok=True)
    for i,(x,y,w,h) in enumerate(panels,1):
        crop = img[y:y+h, x:x+w]
        fn = os.path.join(output_folder, f"panel_{i:02d}.png")
        cv2.imwrite(fn, crop)
        print(f"Saved panel {i}/{len(panels)} → {fn}")

    return panels

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Extract Matplotlib-style subfigures by largest gutter detection"
    )
    p.add_argument("image")
    p.add_argument("-o","--out", default="panels")
    p.add_argument("--bg_tol", type=int, default=30)
    p.add_argument("--gap_frac", type=float, default=0.02)
    p.add_argument("--select_frac", type=float, default=0.5,
                   help="Keep only gaps ≥ select_frac×max_gap")
    p.add_argument("--min_panel_frac", type=float, default=0.005)
    args = p.parse_args()

    extract_matplotlib_subfigures(
        args.image, args.out,
        bg_tol=args.bg_tol,
        gap_frac=args.gap_frac,
        select_frac=args.select_frac,
        min_panel_frac=args.min_panel_frac
    )
