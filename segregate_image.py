import os
import sys
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def remove_small_components(img, area_thresh=200, top_fraction=0.33):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    h_img = img.shape[0]
    debug_img = img.copy()
    removed_count = 0
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < area_thresh and y < h_img * top_fraction:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            removed_count += 1
    if removed_count > 0:
        cv2.imwrite('debug_labels_removed.png', debug_img)
        print(f"OpenCV: removed {removed_count} small components near the top of the image. See debug_labels_removed.png.")
    else:
        print("OpenCV: No small components removed.")
    return img


def remove_single_letter_components(img, area_thresh=100, aspect_min=0.7, aspect_max=1.5, top_fraction=0.25):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    h_img = img.shape[0]
    debug_img = img.copy()
    removed_count = 0
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aspect = w / h if h > 0 else 0
        # Stricter: smaller area, more square, and only near the top
        if 20 < area < area_thresh and aspect_min < aspect < aspect_max and y < h_img * top_fraction:
            print(f"Removing component at ({x},{y},{w},{h}), area={area}, aspect={aspect:.2f}")
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            removed_count += 1
    if removed_count > 0:
        cv2.imwrite('debug_labels_removed.png', debug_img)
        print(f"OpenCV: removed {removed_count} stricter single-letter components. See debug_labels_removed.png.")
    else:
        print("OpenCV: No stricter single-letter components removed.")
    return img


def binarize_image(gray, method=cv2.THRESH_OTSU):
    # Otsu's thresholding for automatic binarization
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + method)
    return binary


def find_valleys(profile, min_gap_size):
    # Find continuous low-value regions (valleys) in the projection profile
    valleys = []
    in_valley = False
    start = 0
    for i, val in enumerate(profile):
        if val == 0 and not in_valley:
            in_valley = True
            start = i
        elif val != 0 and in_valley:
            if i - start >= min_gap_size:
                valleys.append((start, i))
            in_valley = False
    # Handle case where valley goes to the end
    if in_valley and len(profile) - start >= min_gap_size:
        valleys.append((start, len(profile)))
    return valleys


def get_segments(valleys, length):
    # Use valleys to define segments (between valleys)
    segments = []
    prev_end = 0
    for start, end in valleys:
        if prev_end < start:
            segments.append((prev_end, start))
        prev_end = end
    if prev_end < length:
        segments.append((prev_end, length))
    return segments


def remove_target_letter_components(img, area_thresh_min=30, area_thresh_max=300, aspect_min=0.5, aspect_max=2.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use adaptive thresholding for better binarization
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    debug_img = img.copy()
    removed_count = 0
    target_letters = set("ABCDEFabcdef")
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aspect = w / h if h > 0 else 0
        if area_thresh_min < area < area_thresh_max and aspect_min < aspect < aspect_max:
            crop = gray[y:y+h, x:x+w]
            letter = pytesseract.image_to_string(crop, config='--psm 10 -c tessedit_char_whitelist=ABCDEFabcdef').strip()
            print(f"Candidate at ({x},{y},{w},{h}), area={area}, aspect={aspect:.2f}, OCR='{letter}'")
            if len(letter) == 1 and letter in target_letters:
                print(f"Removing letter '{letter}' at ({x},{y},{w},{h}), area={area}, aspect={aspect:.2f}")
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                removed_count += 1
    if removed_count > 0:
        cv2.imwrite('debug_labels_removed.png', debug_img)
        print(f"OpenCV+OCR: removed {removed_count} target letter components. See debug_labels_removed.png.")
    else:
        print("OpenCV+OCR: No target letter components removed.")
    return img


def remove_margin_single_letters(img, area_range=(20, 200), aspect_range=(0.2, 5.0)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    debug_img = img.copy()
    removed_blobs = []  # Store (x, y, w, h, original_pixels)
    removed_count = 0
    for i in range(1, num_labels):
        x, y, w_box, h_box, area = stats[i]
        aspect = w_box / h_box if h_box > 0 else 0
        if area_range[0] < area < area_range[1] and aspect_range[0] < aspect < aspect_range[1]:
            print(f"Removing blob at ({x},{y},{w_box},{h_box}), area={area}, aspect={aspect:.2f}")
            original_pixels = img[y:y+h_box, x:x+w_box].copy()
            removed_blobs.append((x, y, w_box, h_box, original_pixels))
            cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (255, 255, 255), -1)
            cv2.rectangle(debug_img, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
            removed_count += 1
    if removed_count > 0:
        cv2.imwrite('debug_labels_removed.png', debug_img)
        print(f"Removed {removed_count} blobs in full image. See debug_labels_removed.png.")
    else:
        print("No blobs removed in full image.")
    return img, removed_blobs


def filter_close_cuts(cuts, min_dist=10):
    if not cuts:
        return []
    filtered = []
    group = [cuts[0]]
    for c in cuts[1:]:
        if c - group[-1] < min_dist:
            group.append(c)
        else:
            # Keep the cut closest to the center of the group
            center = group[len(group)//2]
            filtered.append(center)
            group = [c]
    # Add the last group
    if group:
        center = group[len(group)//2]
        filtered.append(center)
    return filtered


def segment_image(image_path, output_dir):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return
    # Remove likely single-letter blobs in full image, and get removed blobs
    img_clean, removed_blobs = remove_margin_single_letters(img.copy())
    gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
    binary = binarize_image(gray)

    vertical_profile = np.sum(binary, axis=0)
    horizontal_profile = np.sum(binary, axis=1)
    vertical_profile_bin = (vertical_profile == 0).astype(np.uint8)
    horizontal_profile_bin = (horizontal_profile == 0).astype(np.uint8)

    def find_valley_centers(profile, min_len, axis_name, return_valleys=False):
        centers = []
        valleys = []
        in_valley = False
        start = 0
        for i, val in enumerate(profile):
            if val == 1 and not in_valley:
                in_valley = True
                start = i
            elif val == 0 and in_valley:
                if i - start >= min_len:
                    centers.append((start + i - 1) // 2)
                valleys.append((start, i-1, i-1-start+1))
                in_valley = False
        if in_valley:
            if len(profile) - start >= min_len:
                centers.append((start + len(profile) - 1) // 2)
            valleys.append((start, len(profile)-1, len(profile)-1-start+1))
        print(f"Detected {len(valleys)} {axis_name} valleys:")
        for s, e, l in valleys:
            print(f"  {axis_name} valley: start={s}, end={e}, length={l}")
        if return_valleys:
            return [(s, e, l) for s, e, l in valleys]
        return centers

    min_vgap = max(1, int(img.shape[1] * 3 / 100)) # Current valley min length
    min_hgap = max(1, int(img.shape[0] * 3 / 100))
    # Plot and save projection profiles with valleys
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    axs[0].plot(vertical_profile, color='black', label='Vertical Profile')
    axs[0].set_title('Vertical Projection Profile')
    axs[0].set_xlabel('Column Index')
    axs[0].set_ylabel('Sum')
    # Highlight valleys
    for s, e, l in find_valley_centers(vertical_profile_bin, min_vgap, 'vertical', return_valleys=True):
        axs[0].axvspan(s, e, color='red', alpha=0.3)
    axs[0].legend()

    axs[1].plot(horizontal_profile, color='black', label='Horizontal Profile')
    axs[1].set_title('Horizontal Projection Profile')
    axs[1].set_xlabel('Row Index')
    axs[1].set_ylabel('Sum')
    for s, e, l in find_valley_centers(horizontal_profile_bin, min_hgap, 'horizontal', return_valleys=True):
        axs[1].axvspan(s, e, color='red', alpha=0.3)
    axs[1].legend()
    plt.tight_layout()
    plt.savefig('debug_projection_valleys.png')
    plt.close(fig)

    v_centers = find_valley_centers(vertical_profile_bin, min_vgap, 'vertical')
    h_centers = find_valley_centers(horizontal_profile_bin, min_hgap, 'horizontal')
    v_cuts = [0] + v_centers + [img.shape[1]]
    h_cuts = [0] + h_centers + [img.shape[0]]

    # Filter out cuts that are too close together
    v_min_dist = max(1, int(img.shape[1] * 0.10))
    h_min_dist = max(1, int(img.shape[0] * 0.10))
    v_cuts = filter_close_cuts(sorted(v_cuts), min_dist=v_min_dist)
    h_cuts = filter_close_cuts(sorted(h_cuts), min_dist=h_min_dist)

    # Visualization: draw blobs and cuts on original image
    vis_img = img.copy()
    for x, y, w_box, h_box, _ in removed_blobs:
        cv2.rectangle(vis_img, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
    for x in v_cuts:
        cv2.line(vis_img, (x, 0), (x, img.shape[0]), (255, 0, 0), 1)
    for y in h_cuts:
        cv2.line(vis_img, (0, y), (img.shape[1], y), (255, 0, 0), 1)
    cv2.imwrite('debug_cuts_and_blobs.png', vis_img)
    print("Saved debug_cuts_and_blobs.png with all removed blobs and segmentation cuts.")

    # Restore blobs to the cleaned image before cropping
    img_with_blobs = img_clean.copy()
    for x, y, w_box, h_box, original_pixels in removed_blobs:
        img_with_blobs[y:y+h_box, x:x+w_box] = original_pixels

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for i in range(len(h_cuts) - 1):
        y0, y1 = h_cuts[i], h_cuts[i+1]
        if y1 - y0 <= 1:
            continue
        for j in range(len(v_cuts) - 1):
            x0, x1 = v_cuts[j], v_cuts[j+1]
            if x1 - x0 <= 1:
                continue
            # Expand crop by 5% on each side, clamped to image bounds
            width = x1 - x0
            height = y1 - y0
            x_margin = int(0.08 * width)
            y_margin = int(0.08 * height)
            x0_exp = max(0, x0 - x_margin)
            x1_exp = min(img.shape[1], x1 + x_margin)
            y0_exp = max(0, y0 - y_margin)
            y1_exp = min(img.shape[0], y1 + y_margin)
            # Crop from the cleaned image (without blobs)
            sub_img_clean = img_clean[y0_exp:y1_exp, x0_exp:x1_exp]
            # Skip if the sub_img_clean is mostly white (>=99% white pixels)
            white_ratio = np.mean(sub_img_clean == 255)
            if white_ratio >= 0.99:
                continue
            # Restore blobs to the cleaned image for the final output
            sub_img = img_with_blobs[y0_exp:y1_exp, x0_exp:x1_exp]
            out_path = os.path.join(output_dir, f"subfigure_{i}_{j}.png")
            cv2.imwrite(out_path, sub_img)
            count += 1
    print(f"Saved {count} subfigures to {output_dir}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python segregate_subfigures.py input_image output_dir")
        sys.exit(1)
    image_path = sys.argv[1]
    output_dir = sys.argv[2]
    segment_image(image_path, output_dir)

if __name__ == "__main__":
    main() 