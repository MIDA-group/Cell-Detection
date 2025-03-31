import os
import json
import numpy as np
import pandas as pd
from cellpose import models, io
from scipy.ndimage import center_of_mass
import sys
sys.path.insert(1, '../Utils/')
from json_merger import merge_centroids_json
#from json_merger_oc import merge_centroids_json
from metrics import calculate_metrics
import time 
import torch
import warnings
import tracemalloc
warnings.filterwarnings("ignore")


def process_image_with_cellpose(image_path, model_type='cyto', diameter=None):
    """
    Process a single image using Cellpose.
    
    Args:
        image_path (str): Path to the input image
        model_type (str): Cellpose model type
        diameter (int): Estimated diameter of objects
    
    Returns:
        dict: Processing results including masks, centroids, and image info
    """
    # Read the image
    image = io.imread(image_path)
    
    # Normalize the image
    image = (image - image.min()) / (image.max() - image.min())
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Initialize Cellpose model
    model = models.Cellpose(gpu=True,model_type=model_type)
    
    # Detect nuclei masks
    masks, flows, _, _ = model.eval(image, diameter=diameter, channels=[0, 0])
    
    # Extract centroids
    centroids = [center_of_mass(masks == label) for label in np.unique(masks) if label > 0]
    # Create DataFrame of centroids
    # Note: center_of_mass returns (y,x) coordinates
    centroids_df = pd.DataFrame(centroids, columns=['y', 'x'])
    centroids_df = centroids_df.iloc[:, [1,0]]
    
    # Prepare result dictionary
    result = {
        'filename': os.path.basename(image_path),
        'width': width,
        'height': height,
        'num_objects': len(centroids),
        'centroids': centroids_df.to_dict(orient='list')
    }
    
    return result

def process_image_folder(input_folder, output_json_path, model_type='cyto', diameter=None):
    """
    Process all images in a given folder using Cellpose.
    
    Args:
        input_folder (str): Path to folder containing images
        output_json_path (str): Path to save output JSON file
        model_type (str): Cellpose model type
        diameter (int): Estimated diameter of objects
    """
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # Initialize results list
    results = []
    
    # Iterate through files in the folder
    for filename in os.listdir(input_folder):
        # Check if file is an image
        if os.path.splitext(filename)[1].lower() in image_extensions:
            # Full path to the image
            image_path = os.path.join(input_folder, filename)
            
            try:
                # Process the image
                image_result = process_image_with_cellpose(
                    image_path, 
                    model_type=model_type, 
                    diameter=diameter
                )
                
                # Append to results
                results.append(image_result)
                
                #print(f"Processed: {filename}")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Save results to JSON
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_json_path}")


if __name__ == "__main__":

    input_folder = "/work/marco/SCIA2025/CNSeg/PatchSeg/split1/test-images"
    output_json = "./cellpose_results.json"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    else:
        tracemalloc.start()
    
    
    start = time.time()
    process_image_folder(input_folder, output_json)
    end = time.time()

    if device == "cuda":
        max_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak GPU memory: {max_memory:.2f} MB")
    else: 
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current: {current/1024**2:.2f} MB")
        print(f"Peak: {peak/1024**2:.2f} MB")

    print(f"Execution time: {end - start:.2f} seconds")

    labelme_folder_path = "/work/marco/SCIA2025/CNSeg/PatchSeg/split1/test-labels/"
    output_json_path = "./merged_results_cellpose.json"
    
    merge_centroids_json(output_json, labelme_folder_path, output_json_path)
    for alpha in [0.3,1]:
        calculate_metrics(output_json_path,alpha)