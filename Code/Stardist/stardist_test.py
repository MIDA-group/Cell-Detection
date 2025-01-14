import os
import json
import pandas as pd
from stardist.models import StarDist2D
from skimage import io
from skimage.measure import regionprops
import sys
sys.path.insert(1, '../Utils/')
from json_merger import merge_centroids_json
from metrics import calculate_metrics
import tensorflow as tf
import tracemalloc
import time
import warnings
warnings.filterwarnings("ignore")

# Initialize stardist model
model = StarDist2D.from_pretrained('2D_versatile_he')

def process_image_with_stardist(image_path):
    """
    Process a single image using stardist.
    
    Args:
        image_path (str): Path to the input image
        model_type (str): stardist model type
        
    Returns:
        dict: Processing results including masks, centroids, and image info
    """
    # Read the image
    image = io.imread(image_path)
    
    # Normalize the image
    image = (image - image.min()) / (image.max() - image.min())
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Run the prediction with StarDist on the normalized image
    labels, details = model.predict_instances(image)
    
    # Extract centroids
    centroids = []
    for region in regionprops(labels.astype(int)):
        centroids.append((int(region.centroid[1]), int(region.centroid[0])))

    centroids_df = pd.DataFrame(centroids, columns=['x', 'y'])
    
    # Prepare result dictionary
    result = {
        'filename': os.path.basename(image_path),
        'width': width,
        'height': height,
        'num_objects': len(centroids),
        'centroids': centroids_df.to_dict(orient='list')
    }
    
    return result

def process_image_folder(input_folder, output_json_path):
    """
    Process all images in a given folder using Stardist.
    
    Args:
        input_folder (str): Path to folder containing images
        output_json_path (str): Path to save output JSON file
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
                image_result = process_image_with_stardist(
                    image_path,
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
    input_folder = "/work/marco/SCIA2025/CNSeg/PatchSeg/test-images"
    output_json = "./stardist_results.json"
    
    device = "/GPU:0" if len(tf.config.list_physical_devices('GPU')) > 0 else "/CPU:0"

    if device == "/GPU:0":
        # Clear memory stats and cache
        tf.config.experimental.reset_memory_stats('GPU:0')
    else:
        tracemalloc.start()
    
    start = time.time()
    process_image_folder(input_folder, output_json)
    end = time.time()

    if device == "/GPU:0":
        max_memory = tf.config.experimental.get_memory_info('GPU:0')['peak'] / (1024 * 1024)
        print(f"Peak GPU memory: {max_memory:.2f} MB")
    else: 
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current: {current/1024**2:.2f} MB")
        print(f"Peak: {peak/1024**2:.2f} MB")

    print(f"Execution time: {end - start:.2f} seconds")

    labelme_folder_path = "/work/marco/SCIA2025/CNSeg/PatchSeg/test-labels"
    output_json_path = "./merged_results_stardist.json"
    
    merge_centroids_json(output_json, labelme_folder_path, output_json_path)
    for alpha in [0.3,0.5,1]:
        calculate_metrics(output_json_path,alpha)