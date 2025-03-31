"""
    Regression mini-UNet to predict nucleus-likeness and detect local maxima
    Modified to process JPG images and output JSON results
"""

import argparse
import os
from pathlib import Path
import json
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from skimage.feature import peak_local_max
import sys
sys.path.insert(1, '../Utils/')
#from json_merger import merge_centroids_json
from json_merger_oc import merge_centroids_json
from metrics import calculate_metrics
from unet import UNet
from utils.utils import *
import time
import tracemalloc

def find_spots(img, threshold, min_dist=5):
    xy = np.fliplr(peak_local_max(img, min_distance=min_dist, threshold_abs=threshold, p_norm=2))
    return xy

def process_folder(net, device, input_folder, output_folder, threshold, min_dist, save_masks=False):
    """Process all JPG images in the input folder and save results as JSON"""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all jpg files from input folder
    input_files = list(Path(input_folder).glob('*.jpg')) + list(Path(input_folder).glob('*.jpeg'))
    if not input_files:
        print(f"No JPG images found in {input_folder}")
        return
    
    net.eval()
    count = 0
    
    # Initialize results list for JSON
    results = []
    
    with tqdm(total=len(input_files), unit='img') as pbar:
        for img_path in input_files:
            # Load and preprocess image
            image = Image.open(img_path)
            width, height = image.size
            image = np.array(image)
            if len(image.shape) == 2:  # Convert grayscale to RGB
                image = np.stack((image,) * 3, axis=-1)
            
            # Ensure image is in RGB format
            if image.shape[2] != 3:
                print(f"Skipping {img_path}: expected 3 channels, got {image.shape[2]}")
                continue
                
            # Normalize and convert to tensor
            image = image.transpose((2, 0, 1))  # [H,W,C] -> [C,H,W]
            image = torch.from_numpy(image).float()
            image = image / 255.0  # Normalize to [0,1]
            image = image.unsqueeze(0)
            
            # Process image
            with torch.inference_mode():
                image = image.to(device=device)
                mask_pred = net(image)
            
            # Convert prediction to numpy
            mask = mask_pred.detach().cpu().numpy().squeeze()
            
            # Generate output paths
            out_basename = os.path.splitext(os.path.basename(img_path))[0]
            
            # Save mask if requested
            if save_masks:
                mask_path = os.path.join(output_folder, f"{out_basename}_mask.jpg")
                np2PIL(mask).save(mask_path)
            
            # Find spots
            xy = find_spots(mask, threshold, min_dist)
            count += xy.shape[0] if xy.size > 0 else 0
            
            # Save spots to CSV (keep this for backward compatibility)
            if xy.size > 0:
                csv_path = os.path.join(output_folder, f"{out_basename}_spots.csv")
                np.savetxt(csv_path, xy, fmt="%.1f", delimiter=",", header="x,y")
            
            # Add results to JSON format
            result = {
                'filename': os.path.basename(img_path),
                'width': width,
                'height': height,
                'num_objects': xy.shape[0] if xy.size > 0 else 0,
                'centroids': {
                    'x': xy[:, 0].tolist() if xy.size > 0 else [],
                    'y': xy[:, 1].tolist() if xy.size > 0 else []
                }
            }
            results.append(result)
            
            pbar.update(1)
            pbar.set_postfix_str(f'Total spots={count:5d}')
    
    # Save results to JSON
    json_path = os.path.join(output_folder, 'cnseg_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nProcessed {len(input_files)} images, found {count} spots total")
    return json_path

def get_args():
    parser = argparse.ArgumentParser(description='Predict locations from input images in a folder')
    parser.add_argument('--input', '-i', required=True, help='Input folder containing JPG images')
    parser.add_argument('--output', '-o', required=True, help='Output folder for results')
    parser.add_argument('--model', '-m', default='MODEL.pth', help='File with stored network model')
    parser.add_argument('--threshold', '-t', type=float, default=0.4, help='Threshold for local maxima detection')
    parser.add_argument('--min-dist', '-d', type=float, default=5, help='Minimal distance of local maxima')
    parser.add_argument('--save-masks', '-s', action='store_true', help='Save the predicted regression output')
    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', help='disables macOS GPU training')
    parser.add_argument('--labels-dir', type=str, help='Directory containing ground truth labels for metric calculation')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    # Set up device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    else:
        tracemalloc.start()    
    
    start = time.time()

    # Load model
    net = UNet(n_channels=3, n_classes=1)
    net.load_state_dict(torch.load(args.model, map_location=device, weights_only=False))
    net.to(device=device)
    
    # Process images and get JSON path
    json_path = process_folder(
        net=net,
        device=device,
        input_folder=args.input,
        output_folder=args.output,
        threshold=args.threshold,
        min_dist=args.min_dist,
        save_masks=args.save_masks
    )
    
    end = time.time()

    if device == "cuda":
        max_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak GPU memory: {max_memory:.2f} MB")
    else: 
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current: {current/1024**2:.2f} MB")
        print(f"Peak: {peak/1024**2:.2f} MB")

    print(f"Execution time: {end - start:.2f} seconds")

    # Calculate metrics if labels directory is provided
    if args.labels_dir:
        # Merge results with ground truth
        merged_json_path = os.path.join(args.output, 'merged_results_cnseg.json')
        merge_centroids_json(json_path, args.labels_dir, merged_json_path)
        
        # Calculate metrics for different alpha values
        for alpha in [0.3, 0.5, 1]:
            calculate_metrics(merged_json_path, alpha)