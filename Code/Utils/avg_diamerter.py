import json
import math
import os
from pathlib import Path
import numpy as np

def calculate_polygon_area(points):
    """Calculate area of polygon using shoelace formula"""
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return 0.5 * abs(sum(i * j for i, j in zip(x, y[1:] + [y[0]])) - 
                    sum(i * j for i, j in zip(x[1:] + [x[0]], y)))

def calculate_radius(area):
    """Calculate radius of circle with equivalent area"""
    return math.sqrt(area / math.pi)

def analyze_shapes(folder_path):
    """Analyze all JSON files in folder for shape areas and radii"""
    areas = []
    radii = []
    
    for file_path in Path(folder_path).glob('*.json'):
        with open(file_path) as f:
            data = json.load(f)
            
        for shape in data.get('shapes', []):
            if shape.get('shape_type') == 'polygon':
                area = calculate_polygon_area(shape['points'])
                radius = calculate_radius(area)
                areas.append(area)
                radii.append(radius)
    
    return {
        'average_area': np.mean(areas),
        'average_radius': np.mean(radii),
        'num_shapes': len(areas)
    }

results = analyze_shapes("/work/marco/SCIA2025/CNSeg/PatchSeg/train-labels")
print(f"Average area: {results['average_area']:.2f} square pixels")
print(f"Average radius: {results['average_radius']:.2f} pixels")
print(f"Number of shapes analyzed: {results['num_shapes']}")