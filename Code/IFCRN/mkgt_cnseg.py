import os
from pathlib import Path
import argparse
import json
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

def calculate_geometric_centroid(points):
    points = np.array(points)
    
    # Ensure the polygon is closed
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    
    # Number of vertices
    n = len(points) - 1
    
    # Initialize area and centroids
    area = 0
    cx, cy = 0, 0
    
    # Calculate centroid using the signed area method
    for i in range(n):
        # Current point and next point
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        
        # Calculate signed area contribution
        signed_area = x1 * y2 - x2 * y1
        area += signed_area
        
        # Centroid contribution
        cx += (x1 + x2) * signed_area
        cy += (y1 + y2) * signed_area
    
    # Normalize by 6 * total area
    area *= 0.5
    if area != 0:
        cx /= (6 * area)
        cy /= (6 * area)
    else:
        # Fallback to mean if area is zero
        cx, cy = np.mean(points[:n], axis=0)
    
    return [cx, cy]

def get_args():
    parser = argparse.ArgumentParser(description='Generate masks from images using annotation files')
    parser.add_argument('input_dir', metavar='INPUT_DIR', help='Input directory containing JPG images')
    parser.add_argument('annotations_dir', metavar='ANNOTATIONS_DIR', help='Directory containing annotation JSON files')
    parser.add_argument('--bin', '-b', metavar='B', type=int, help='bin-size', default=4, required=False)
    parser.add_argument('--sigma', '-s', metavar='SIGMA', type=float, help='Gaussian blur sigma', default=3.0, required=False)
    parser.add_argument('--output', '-o', metavar='OUTDIR', help='Directory where to write masks', required=True)
    parser.add_argument('--prefix', '-p', metavar='PREFIX', help='Filename prefix', default="", required=False)
    parser.add_argument('--class', '-c', dest='mclass', metavar='CLASS', nargs='+', help='Classes to include as positive', default=['nucleus'], required=False)
    return parser.parse_args()

def process_image_with_annotations(image_path, annotations_dir, args):
    # Find corresponding JSON file in annotations directory
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(annotations_dir, f"{image_name}.json")
    
    if not os.path.exists(json_path):
        print(f"Warning: No annotation file found for {image_path}")
        return
    
    # Load image and annotations
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Process annotations to get centroids
    markers = []
    
    # Process shapes from LabelMe format
    for shape in data['shapes']:
        if shape['label'] in args.mclass:
            points = np.array(shape['points'])
            centroid = calculate_geometric_centroid(points)
            markers.append(centroid)
    
    if not markers:
        print(f"No valid annotations found for {image_path}")
        return
    
    # Create mask for the entire image
    points = np.array(markers)
    points = points / args.bin  # Scale points according to bin size
    
    # Create the mask with scaled dimensions
    mask_width = int(np.ceil(img_width / args.bin))
    mask_height = int(np.ceil(img_height / args.bin))
    mask = np.zeros((mask_height, mask_width))
    
    # Place markers on the mask
    for point in points:
        x, y = point.astype(int)
        if 0 <= y < mask_height and 0 <= x < mask_width:  # Check bounds
            mask[y, x] = 1
    
    # Apply Gaussian blur
    peakval = gaussian_filter(np.ones([1,1]), sigma=args.sigma, mode='constant').max()
    mask = gaussian_filter(mask, sigma=args.sigma, mode='constant')
    mask /= peakval
    mask = np.clip(mask, 0, 1)
    
    # Save mask
    mask_filename = f'{args.prefix}{image_name}_mask.jpg'
    mask_path = os.path.join(args.output, mask_filename)
    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

def main():
    args = get_args()
    
    # Create output directory
    Path(args.output).mkdir(exist_ok=True)
    
    # Process each image in the input directory
    for file in os.listdir(args.input_dir):
        if file.lower().endswith(('.jpg', '.jpeg')):
            image_path = os.path.join(args.input_dir, file)
            print(f'Processing {file}...')
            process_image_with_annotations(image_path, annotations_dir=args.annotations_dir, args=args)

if __name__ == '__main__':
    main()