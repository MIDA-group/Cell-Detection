import tensorflow as tf
import os
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(1, '../Utils/')
#from json_merger import merge_centroids_json
from json_merger_oc import merge_centroids_json
from metrics import calculate_metrics
from model_builder import buildModel_U_net
import tracemalloc
import time
import warnings
warnings.filterwarnings("ignore")
from skimage import filters, measure
import cv2

IMG_WIDTH = 256
IMG_HEIGHT = 256

model = buildModel_U_net(input_dim=(IMG_WIDTH, IMG_HEIGHT, 3))
model.load_weights('./saved_models/split4_oc_best_weights.weights.h5')


def load_single_image(image_path, img_width=IMG_WIDTH, img_height=IMG_HEIGHT):
    """
    Load and preprocess a single image for inference

    Args:
        image_path (str): Path to the input image
        img_width (int): Target width for resizing
        img_height (int): Target height for resizing

    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Read the image
    img = cv2.imread(image_path)

    # Resize the image
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)

    # Normalize the image (zero mean, unit variance)
    img_normalized = (img - np.mean(img)) / np.std(img)

    # Add batch dimension
    img_normalized = np.expand_dims(img_normalized, axis=0)

    return img_normalized

def detect_single_image(model, image_path, threshold=0.65):
    """
    Run inference on a single image

    Args:
        model: Trained Keras model
        image_path (str): Path to the input image
        threshold (float): Threshold for binary segmentation

    Returns:
        numpy.ndarray: Predicted mask
    """
    # Preprocess the image
    processed_image = load_single_image(image_path)

    # Run inference
    start = time.time()
    prediction = model.predict(processed_image)

    # Post-process the prediction
    preds = np.where(prediction > 0, prediction / 100, prediction)
    preds = (preds + 1) / 2
    preds_thresholded = (preds > threshold).astype(np.uint8)

    return np.squeeze(preds_thresholded)

def process_image_with_fcrn(image_path):
    """
    Process a single image using fcrn.
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        dict: Processing results including centroids and image info
    """ 

    # Run inference
    masks = detect_single_image(model, image_path, threshold=0.65)
 
    # Using a default thresholding method
    thresh = filters.threshold_otsu(masks)
    binary = masks > thresh

    # Measure properties of labeled regions
    labels = measure.label(binary)
    properties = measure.regionprops(labels)

    # Extract measurements
    centroids = []
    for prop in properties:
            centroids.append(
                (prop.centroid[1],prop.centroid[0])
            )

    centroids_df = pd.DataFrame(centroids, columns=['x', 'y'])
    
    # Prepare result dictionary
    result = {
        'filename': os.path.basename(image_path),
        'width': IMG_WIDTH,
        'height': IMG_HEIGHT,
        'num_objects': len(centroids),
        'centroids': centroids_df.to_dict(orient='list')
    }
    
    return result

def process_image_folder(input_folder, output_json_path):
    """
    Process all images in a given folder using fcrn.
    
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
                image_result = process_image_with_fcrn(
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
    input_folder = "/work/marco/SCIA2025/OC/split4/test-images"
    output_json = "./fcrn_results.json"
    
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

    labelme_folder_path = "/work/marco/SCIA2025/OC/split4/test-labels"
    output_json_path = "./merged_results_fcrn.json"
    
    merge_centroids_json(output_json, labelme_folder_path, output_json_path)
    for alpha in [0.3,0.5,1]:
        calculate_metrics(output_json_path,alpha)