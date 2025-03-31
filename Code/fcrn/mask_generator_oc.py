import numpy as np
import cv2
from pathlib import Path
import json
import sys
sys.path.insert(1, '../Utils/')

# Set parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256

class MaskGenerator:
    def __init__(self, 
                 image_dir: str,
                 annotation_dir: str,
                 output_dir: str,
                 removed_dir: str,
                 img_width: int = IMG_WIDTH,
                 img_height: int = IMG_HEIGHT):
        """
        Initialize the mask generator
        
        Args:
            image_dir: Directory containing the images
            annotation_dir: Directory containing LabelMe annotation files
            output_dir: Directory to save the generated masks
            img_width: Width for point masks
            img_height: Height for point masks
        """
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.output_dir = Path(output_dir)
        self.img_width = img_width
        self.img_height = img_height
        self.removed_dir = removed_dir

        # Create output directories
        self.mask_dirs = {
            'disk': self.output_dir,
        }
        
        # Create all directories
        for dir_path in self.mask_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def dist(point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt(sum((x1 - x2) ** 2 for x1, x2 in zip(point1, point2)))


    def load_annotations(self, image_name):
        """Load annotations from LabelMe JSON file and calculate centroids"""
        json_path = self.annotation_dir / f"{image_name}.json"
        if not json_path.exists():
            print(f"Warning: No annotation file found for {image_name}")
            return []
        
        with open(json_path) as f:
            data = json.load(f)
        
        centroids = []
        for shape in data.get('shapes', []):
            points = shape.get('points', [])
            if points:
                if len(points) == 1:
                    x, y = points[0]
                    centroids.append([x, y])
                else:
                    points = np.array(points)
                    for i in range(len(points) - 1):
                        x, y = points[i]
                        centroid = [x, y]
                        centroids.append(centroid)
        
        return centroids

    def generate_disk_mask(self, image_path, centroids, radius):
        """Generate mask with disks at each nucleus centroid"""
        img = cv2.imread(str(image_path))
        mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        
        for centroid in centroids:
            center_x = int(centroid[0])
            center_y = int(centroid[1])
            
            x_min = max(0, center_x - radius)
            x_max = min(img.shape[1], center_x + radius + 1)
            y_min = max(0, center_y - radius)
            y_max = min(img.shape[0], center_y + radius + 1)
            
            for x in range(x_min, x_max):
                for y in range(y_min, y_max):
                    if self.dist([x, y], [center_x, center_y]) <= radius:
                        mask[y, x] = 255
        
        return mask

    def save_mask(self, mask, image_name, mask_type):
        
        cv2.imwrite(
            str(self.mask_dirs[mask_type] / f"{image_name}_mask.jpg"),
            mask,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        )

    def generate_masks(self, radius=15):
        """Generate both disk and point masks for all images"""
        supported_formats = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in self.image_dir.iterdir() 
                      if f.suffix.lower() in supported_formats]
        
        for img_path in image_files:
            image_name = img_path.stem
            
            # Load annotations and calculate centroids
            centroids = self.load_annotations(image_name)
            if not centroids:
                print(f"No nuclei found in {image_name}")
                continue

            # Generate disk mask
            disk_mask = self.generate_disk_mask(img_path, centroids, radius)
            self.save_mask(disk_mask, image_name, 'disk')
   
        print("Mask generation completed!")



if __name__ == '__main__':

    image_dir = "/work/marco/SCIA2025/OC/split4/train-images/"         
    annotation_dir = "/work/marco/SCIA2025/OC/split4/train-labels/"
    output_dir = "/work/marco/SCIA2025/OC/split4/train-masks"
    removed_dir = "/work/marco/SCIA2025/OC/train-removed"

    generator = MaskGenerator(
        image_dir = image_dir,
        annotation_dir=annotation_dir,
        output_dir=output_dir,
        removed_dir=removed_dir
    )
    
    generator.generate_masks(radius=22)