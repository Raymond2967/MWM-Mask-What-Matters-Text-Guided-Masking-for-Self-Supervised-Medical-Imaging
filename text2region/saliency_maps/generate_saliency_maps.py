import warnings
warnings.filterwarnings('ignore')
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import itertools
import torch
import json
import random
import numpy as np
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast
from PIL import Image
from scripts.methods import vision_heatmap_iba

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def calculate_dice_coefficient(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    dice_coefficient = (2.0 * intersection) / (mask1.sum() + mask2.sum())
    return dice_coefficient

def evaluate_on_sample(model, processor, tokenizer, text, image_paths, args):
    dice_scores = []
    for image_id in tqdm(image_paths):
        try:
            image = Image.open(f"{args.val_path}/{image_id}").convert('RGB')
        except:
            continue
        image_feat = processor(images=image, return_tensors="pt")['pixel_values'].to(args.device)
        text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)]).to(args.device)
        vmap = vision_heatmap_iba(text_ids, image_feat, model, args.vlayer, args.vbeta, args.vvar, ensemble=args.ensemble, progbar=False)
        gt_path = args.val_path.replace("images", "masks")
        gt_mask = np.array(Image.open(f"{gt_path}/{image_id}").convert("L"))
        vmap_resized = cv2.resize(np.array(vmap), (gt_mask.shape[1], gt_mask.shape[0]))
        cam_img = vmap_resized > 0.3
        dice_score = calculate_dice_coefficient(gt_mask.astype(bool), cam_img.astype(bool))
        dice_scores.append(dice_score)
    average_dice = np.mean(dice_scores)
    return average_dice

def hyper_opt(model, processor, tokenizer, text, args):
    print("Running Hyperparameter Optimization ...")
    vbeta_list = [0.1, 1.0, 2.0]
    vvar_list = [0.1, 1.0, 2.0]
    layers_list = [7, 8, 9]
    hyperparameter_combinations = list(itertools.product(vbeta_list, vvar_list, layers_list))
    all_image_ids = sorted(os.listdir(args.val_path))
    results = []
    for combo in hyperparameter_combinations:
        vbeta, vvar, layer = combo
        args.vbeta = vbeta
        args.vvar = vvar
        args.vlayer = layer
        sample_dice_scores = []
        print(f"Evaluating combination: vbeta={vbeta}, vvar={vvar}, layer={layer}")
        for i in range(3):
            random.seed(i)
            sampled_images = random.sample(all_image_ids, 1)
            avg_dice = evaluate_on_sample(model, processor, tokenizer, text, sampled_images, args)
            sample_dice_scores.append(avg_dice)
            print(f"  Sample {i+1}: Average Dice Score = {avg_dice}")
        mean_dice = np.mean(sample_dice_scores)
        results.append({
            'vbeta': vbeta,
            'vvar': vvar,
            'vlayer': layer,
            'average_dice': mean_dice
        })
        print(f"Mean Dice Score for this combination: {mean_dice}\n")
    results_df = pd.DataFrame(results)
    best_combo = results_df.loc[results_df['average_dice'].idxmax()]
    print("Best Hyperparameter Combination:")
    print(best_combo)
    print("\n")
    return best_combo

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # Auto-select device if needed
    if args.device not in ("cpu", "cuda"):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        # If user set cuda but it's unavailable, fall back gracefully
        if args.device == "cuda" and not torch.cuda.is_available():
            args.device = "cpu"
    print("Loading models ...")
    if args.model_name == "BiomedCLIP" and args.finetuned:
        model_path = "./saliency_maps/model"
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(args.device)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif(args.model_name == "BiomedCLIP" and not args.finetuned):
        model = AutoModel.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True).to(args.device)
        processor = AutoProcessor.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
    elif(args.model_name == "CLIP" and not args.finetuned):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(args.device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16")
    elif(args.model_name == "CLIP" and args.finetuned):
        model_path = "./model"
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(args.device)
        # Try to load matching processor/tokenizer for the finetuned CLIP
        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16")
    if(not args.reproduce):
        text = str(input("Enter the text: "))
    if(args.hyper_opt):
        best_combo = hyper_opt(model, processor, tokenizer, text, args)
        args.vbeta = best_combo['vbeta']
        args.vvar = best_combo['vvar']
        args.vlayer = int(best_combo['vlayer'])
    print("Generating Saliency Maps ...")
    if(not os.path.exists(args.output_path)):
        os.makedirs(args.output_path)
    existing_outputs = set(os.listdir(args.output_path))
    for image_id in tqdm(sorted(os.listdir(args.input_path))):
        if image_id in existing_outputs:
            continue
        try:
            image = Image.open(f"{args.input_path}/{image_id}").convert('RGB')
        except:
            print(f"Unable to load image at {image_id}", flush=True)
            continue
        if(args.reproduce):
            with open(args.json_path) as json_file:
                json_decoded = json.load(json_file)
            text = json_decoded[image_id]
        image_feat = processor(images=image, return_tensors="pt")['pixel_values'].to(args.device)
        text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)]).to(args.device)
        vmap = vision_heatmap_iba(text_ids, image_feat, model, args.vlayer, args.vbeta, args.vvar, ensemble=args.ensemble, progbar=False)
        img = np.array(image)
        vmap = cv2.resize(np.array(vmap), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f"{args.output_path}/{image_id}", (vmap * 255).astype(np.uint8))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('M2IB argument parser')
    parser.add_argument('--input-path', required=True, default="data/input_images", type=str)
    parser.add_argument('--output-path', required=True, default="saliency_map_outputs", type=str)
    parser.add_argument('--val-path', type=str, default="data/val_images")
    parser.add_argument('--vbeta', type=float, default=0.1)
    parser.add_argument('--vvar', type=float, default=1.0)
    parser.add_argument('--vlayer', type=int, default=7)
    parser.add_argument('--tbeta', type=float, default=0.3)
    parser.add_argument('--tvar', type=float, default=1)
    parser.add_argument('--tlayer', type=int, default=9)
    parser.add_argument('--model-name', type=str, default="BiomedCLIP")
    parser.add_argument('--finetuned', action='store_true')
    parser.add_argument('--hyper-opt', action='store_true')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--json-path', type=str, default="busi.json")
    parser.add_argument('--reproduce', action='store_true')
    args = parser.parse_args()
    main(args)
    print("Saliency Map Generation Done!")
