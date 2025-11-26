import numpy as np
import cv2
import os
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm
import argparse
import json

# Optional dependency: SurfaceDice. Provide safe fallbacks if not available.
try:
    from SurfaceDice import (
        compute_surface_distances,
        compute_surface_dice_at_tolerance,
        compute_dice_coefficient,
    )
    _HAS_SURFACE_DICE = True
except Exception:
    _HAS_SURFACE_DICE = False

    def compute_dice_coefficient(gt, pred):
        gt = gt.astype(bool)
        pred = pred.astype(bool)
        inter = np.logical_and(gt, pred).sum()
        denom = gt.sum() + pred.sum()
        return (2.0 * inter / denom) if denom > 0 else 0.0

    def compute_surface_distances(gt, pred, spacing):
        # Fallback stub: return None-like placeholder
        return None

    def compute_surface_dice_at_tolerance(surface_distances, tolerance):
        # Fallback stub: without SurfaceDice, return 0.0 to avoid crash
        return 0.0

join = os.path.join
basename = os.path.basename

parser = argparse.ArgumentParser()
parser.add_argument('--gt_path', type=str, required=True)
parser.add_argument('--seg_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--expanded-path', type=str, default='')
parser.add_argument('--eval-type', type=str, required=True,
                    choices=['sam_output', 'coarse_output', 'coarse_expanded', 'sam_expanded'])
args = parser.parse_args()

gt_path = args.gt_path
seg_path = args.seg_path
expanded_path = args.expanded_path
output_dir = args.output_dir
eval_type = args.eval_type

os.makedirs(output_dir, exist_ok=True)

filenames = os.listdir(seg_path)
filenames = [x for x in filenames if x.lower().endswith(('.png', '.jpg', '.jpeg'))]
filenames = [x for x in filenames if os.path.exists(join(seg_path, x))]
filenames.sort()

metric_values = {
    'DSC': [], 'NSD': [], 'Coverage': [], 'Recall': [],
    'Occlusion_Coverage': [], 'Occlusion_Recall': []
}
valid_samples = 0

def compute_metrics(gt, pred):
    if pred is None or gt is None:
        return (0.0,) * 4
    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()
    dsc = compute_dice_coefficient(gt, pred)
    nsd = compute_surface_dice_at_tolerance(
        compute_surface_distances(gt[..., None], pred[..., None], [1, 1, 1]), 2
    )
    coverage = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return dsc, nsd, coverage, recall

for name in tqdm(filenames, desc=f"Evaluating {eval_type}"):
    gt_mask = cv2.imread(join(gt_path, name), cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        continue
    seg_mask = cv2.imread(join(seg_path, name), cv2.IMREAD_GRAYSCALE)
    if seg_mask is None:
        continue
    seg_mask = cv2.resize(seg_mask, (gt_mask.shape[1], gt_mask.shape[0]))
    gt_data = cv2.threshold(gt_mask, 200, 255, cv2.THRESH_BINARY)[1].astype(bool)
    seg_data = cv2.threshold(seg_mask, 200, 255, cv2.THRESH_BINARY)[1].astype(bool)
    dsc, nsd, coverage, recall = compute_metrics(gt_data, seg_data)
    metric_values['DSC'].append(dsc)
    metric_values['NSD'].append(nsd)
    metric_values['Coverage'].append(coverage)
    metric_values['Recall'].append(recall)
    if eval_type.endswith('_expanded') and expanded_path:
        if os.path.exists(join(expanded_path, name)):
            ex_mask = cv2.imread(join(expanded_path, name), cv2.IMREAD_GRAYSCALE)
            if ex_mask is not None:
                ex_mask = cv2.resize(ex_mask, (gt_mask.shape[1], gt_mask.shape[0]))
                ex_mask = cv2.threshold(ex_mask, 200, 255, cv2.THRESH_BINARY)[1].astype(bool)
                _, _, occlusion_coverage, occlusion_recall = compute_metrics(gt_data, ex_mask)
            else:
                occlusion_coverage, occlusion_recall = 0.0, 0.0
        else:
            occlusion_coverage, occlusion_recall = 0.0, 0.0
        metric_values['Occlusion_Coverage'].append(occlusion_coverage)
        metric_values['Occlusion_Recall'].append(occlusion_recall)
    valid_samples += 1

if valid_samples > 0:
    avg_metrics = {
        key: np.mean(values) for key, values in metric_values.items() if values
    }
    csv_data = {
        'Metric': list(avg_metrics.keys()),
        'Average': [round(float(v), 4) for v in avg_metrics.values()]
    }
    pd.DataFrame(csv_data).to_csv(
        join(output_dir, f"{basename(seg_path)}_{eval_type}_summary.csv"),
        index=False
    )
    result = {
        "dataset": f"{basename(seg_path)}_{eval_type}",
        "total_samples": valid_samples,
        "metrics": {k: round(float(v), 4) for k, v in avg_metrics.items()}
    }
    with open(join(output_dir, f"{basename(seg_path)}_{eval_type}_summary.json"), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n[{eval_type.upper()}] Evaluation results (N={valid_samples})")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
else:
    print("No valid samples for evaluation")

print(f"Results saved to: {output_dir}")
