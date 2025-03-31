import os
import json
import numpy as np
import pandas as pd
import torch
from skimage import io, transform
from skimage.measure import regionprops
from segment_anything import SamAutomaticMaskGenerator
from segment_anything.build_sam import build_sam_vit_b
import sys
sys.path.insert(1, '../Utils/')
from json_merger import merge_centroids_json
from metrics import calculate_metrics
import time
import tracemalloc
import warnings
warnings.filterwarnings("ignore")


import torch.nn.functional as F
from segment_anything import sam_model_registry


if torch.cuda.is_available():
    device = torch.device("cuda")

model_checkpoint_path = './medsam_vit_b.pth'
medsam_model = sam_model_registry["vit_b"](checkpoint=model_checkpoint_path)
medsam_model.to(device)
medsam_model.eval()

bounding_box = [0,0,512,512]

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):

    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=None,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

def process_image_with_sam(image_path):
    """
    Process a single image using MedSAM.
    
    Args:
        image_path (str): Path to the input image
    
    Returns:
        dict: Processing results including masks, centroids, and image info
    """
    # Read the image
    image = io.imread(image_path)
    
    # Normalize the image
    image = (image - image.min()) / (image.max() - image.min())
    image_np = np.array(image)
    #image_np = np.float32(image_np)
    H, W, _ = image_np.shape
    
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  
    
    # Get image dimensions
    height, width = image.shape[:2]
    image_np = transform.resize(image_np, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    image_tensor = (torch.tensor(image_np).float().permute(2, 0, 1).unsqueeze(0).to(device))

    box_np = np.array([bounding_box])
    box_1024 = box_np / np.array([W, H, W, H]) * 1024

    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(image_tensor)

    masks = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
    labeled_image = np.zeros(image.shape[:2], dtype=np.int32)
    labeled_image[masks > 0] = 1
    labeled_image = transform.resize(labeled_image, (512, 512), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)

    # Extract centroids
    centroids = []
    for region in regionprops(labeled_image):
        centroids.append((region.centroid[1], region.centroid[0]))

    centroids_df = pd.DataFrame(centroids, columns=['x', 'y'])

    # Prepare result dictionary
    result = {
        'filename': os.path.basename(image_path),
        'width': width,
        'height': height,
        'num_objects': len(centroids),
        'centroids': centroids_df.to_dict(orient='list')
    }
    torch.cuda.empty_cache()
    return result

def process_image_folder(input_folder, output_json_path):
    """
    Process all images in a given folder using MedSAM.
    
    Args:
        input_folder (str): Path to folder containing images
        output_json_path (str): Path to save output JSON file
        model_type (str): sam model type
        diameter (int): Estimated diameter of objects
    """
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # Initialize results list
    results = []
    
    counter = 0
    # Iterate through files in the folder
    for filename in os.listdir(input_folder):
        # Check if file is an image
        if counter == 5:
            break
        else:
            if os.path.splitext(filename)[1].lower() in image_extensions:
                # Full path to the image
                image_path = os.path.join(input_folder, filename)
                
                try:
                    # Process the image
                    image_result = process_image_with_sam(
                        image_path,
                    )
                    
                    # Append to results
                    results.append(image_result)
                    counter = counter + 1
                    print(f"Processed: {filename}")
                
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    
    # Save results to JSON
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_json_path}")


if __name__ == "__main__":
    input_folder = "/work/marco/SCIA2025/CNSeg/PatchSeg/split1/test-images"
    output_json = "./MedSAM_results.json"
    
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

    labelme_folder_path = "/work/marco/SCIA2025/CNSeg/PatchSeg/split1/test-labels"
    output_json_path = "./merged_results_MedSAM.json"
    
    merge_centroids_json(output_json, labelme_folder_path, output_json_path)
    for alpha in [0.3,0.5,1]:
        calculate_metrics(output_json_path,alpha)