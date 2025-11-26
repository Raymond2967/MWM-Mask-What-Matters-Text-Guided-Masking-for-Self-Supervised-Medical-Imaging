import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse  

def expand_mask(mask, ratio=1.5):
    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    expanded_mask = np.zeros_like(mask)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        center_x, center_y = x + w // 2, y + h // 2
        new_w, new_h = int(w * ratio), int(h * ratio)
        x1 = max(0, center_x - new_w // 2)
        y1 = max(0, center_y - new_h // 2)
        x2 = min(mask.shape[1], center_x + new_w // 2)
        y2 = min(mask.shape[0], center_y + new_h // 2)
        expanded_mask[y1:y2, x1:x2] = 255

    return expanded_mask

def get_common_files(*dirs):
    common_files = set(os.listdir(dirs[0]))
    for d in dirs[1:]:
        if not os.path.exists(d):
            print(f"[INFO] Directory does not exist: {d}")
            return []
        common_files &= set(os.listdir(d))
    return list(common_files)

def main(args):
    os.makedirs(args.output_path, exist_ok=True)

    all_files = get_common_files(args.sam_path)
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in tqdm(image_files, desc="Processing SAM masks"):
        sam_mask_path = os.path.join(args.sam_path, filename)
        sam_mask = cv2.imread(sam_mask_path, 0)
        if sam_mask is None:
            continue

        if args.generate_expanded:
            expanded_mask = expand_mask(sam_mask, ratio=args.expand_ratio)
            cv2.imwrite(os.path.join(args.output_path, filename), expanded_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Postprocess SAM masks to generate expanded bounding boxes")
    parser.add_argument("--sam-path", type=str, required=True, help="Path to SAM output masks")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save expanded masks")
    parser.add_argument("--expand-ratio", type=float, default=1.5, help="Ratio to expand the mask")
    parser.add_argument("--generate-expanded", action="store_true", help="Whether to generate expanded masks")

    args = parser.parse_args()
    main(args)
