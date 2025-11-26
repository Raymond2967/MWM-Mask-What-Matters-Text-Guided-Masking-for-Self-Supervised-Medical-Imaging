import os
import argparse
import requests
import torch
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
from random import sample
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast
from PIL import Image
from pytorch_grad_cam.metrics.cam_mult_image import DropInConfidence, IncreaseInConfidence
sys.path.append(os.getcwd())
from scripts.utils import ImageFeatureExtractor, TextFeatureExtractor, CosSimilarity
from scripts.methods import vision_heatmap_iba, text_heatmap_iba
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_metrics(image_feat, vmap, text_ids, tmap, model):
    results = {}
    with torch.no_grad():
        vtargets = [CosSimilarity(model.get_text_features(text_ids).to(device))]
        ttargets = [CosSimilarity(model.get_image_features(image_feat).to(device))]
        text_ids = text_ids[:,1:-1]
        tmap = np.expand_dims(tmap, axis=0)[:,1:-1]
        tmap = tmap > np.percentile(tmap, 50)
        results['vdrop'] = DropInConfidence()(image_feat, vmap, vtargets, ImageFeatureExtractor(model))[0][0]*100
        results['vincr'] = IncreaseInConfidence()(image_feat, vmap, vtargets, ImageFeatureExtractor(model))[0][0]*100
        results['tdrop'] = DropInConfidence()(text_ids, tmap, ttargets, TextFeatureExtractor(model))[0][0]*100
        results['tincr'] = IncreaseInConfidence()(text_ids, tmap, ttargets, TextFeatureExtractor(model))[0][0]*100
    return results

def main(args):
    print("Loading models ...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    df = pd.read_csv(args.data_path,sep='\t')
    data = list(df.itertuples(index=False))
    all_results = []
    print("Evaluating ...")
    for text, image_path in tqdm(sample(data, args.samples)):
        try:
            image = Image.open(requests.get(image_path, stream=True, timeout=5).raw) if 'http' in image_path else Image.open(image_path).convert('RGB')
        except:
            print(f"Unable to load image at {image_path}", flush=True)
            continue
        image_feat = processor(images=image, return_tensors="pt")['pixel_values'].to(device)
        text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)]).to(device)
        vmap = vision_heatmap_iba(text_ids, image_feat, model, args.vlayer, args.vbeta, args.vvar, progbar=False)
        tmap = text_heatmap_iba(text_ids, image_feat, model, args.tlayer, args.tbeta, args.tvar, progbar=False)
        results = get_metrics(image_feat, vmap, text_ids, tmap, model)
        results['image'] = image_path
        results['text'] = text
        all_results.append(results)
    all_results = pd.DataFrame(all_results)
    print(all_results.mean(axis=0), flush=True)
    all_results.to_csv(args.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('M2IB argument parser')
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--output_path', required=True, type=str)
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--vbeta', type=int, default=0.1)
    parser.add_argument('--vvar', type=int, default=1)
    parser.add_argument('--vlayer', type=int, default=9)
    parser.add_argument('--tbeta', type=int, default=0.1)
    parser.add_argument('--tvar', type=int, default=1)
    parser.add_argument('--tlayer', type=int, default=9)
    args = parser.parse_args()
    main(args)
