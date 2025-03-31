import json
import os
import pandas as pd
import numpy as np

def extract_centroids_from_labelme_json(json_path):
    """
    Extract centroids from a LabelMe annotation JSON file.
    
    Args:
        json_path (str): Path to the LabelMe JSON file
    
    Returns:
        pd.DataFrame: Centroids extracted from the shapes
    """
    with open(json_path, 'r') as f:
        labelme_data = json.load(f)
    
    # Extract centroids from polygon points
    centroids = []
    for shape in labelme_data.get('shapes', []):
        points = shape.get('points', [])

        if points:
            if len(points) == 1:
                x, y = points[0]
                centroids.append([x, y])
                # For polygon or line shapes, keep the previous logic
            else:
                points = np.array(points)
                for i in range(len(points) - 1):
                    x, y = points[i]
                    centroid = [x, y]
                    centroids.append(centroid)
    
    if centroids:
        centroids_df = pd.DataFrame(centroids, columns=['x', 'y'])
        return centroids_df
    
    return pd.DataFrame(columns=['x', 'y'])

def merge_centroids_json(_json_path, labelme_folder_path, output_json_path):
    """
    Merge centroids from  results and LabelMe ground truth JSONs.
    
    Args:
        _json_path (str): Path to the  results JSON
        labelme_folder_path (str): Path to the folder containing LabelMe JSONs
        output_json_path (str): Path to save the merged results JSON
    """
    with open(_json_path, 'r') as f:
        _results = json.load(f)

    merged_results = []
    
    for entry in _results:
        filename = entry.get('filename', '')
        
        labelme_json_path = os.path.join(
            labelme_folder_path, 
            os.path.splitext(filename)[0] + '.json'
        )
        
        if not os.path.exists(labelme_json_path):
            print(f"Warning: No LabelMe JSON found for {filename}")
            continue
        
        gt_centroids = extract_centroids_from_labelme_json(labelme_json_path)
        
        pred_centroids_dict = entry.get('centroids', {})
        if pred_centroids_dict:
            pred_centroids = pd.DataFrame({
                'x': pred_centroids_dict.get('x', []),
                'y': pred_centroids_dict.get('y', [])
            })
        else:
            pred_centroids = pd.DataFrame(columns=['x', 'y'])
        
        # Create merged entry
        merged_entry = {
            'width': entry.get('width', 0),
            'height': entry.get('height', 0),
            'gt_centroids': gt_centroids,
            'pred_centroids': pred_centroids
        }
        
        merged_results.append(merged_entry)
    
    class PandasEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='list')
            return json.JSONEncoder.default(self, obj)
    
    # Save merged results to JSON
    with open(output_json_path, 'w') as f:
        json.dump(merged_results, f, indent=4, cls=PandasEncoder)
    
    print(f"Merged results saved to {output_json_path}")