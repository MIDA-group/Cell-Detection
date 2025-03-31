from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2

class BasicDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform=None,
        file_list=None
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        
        if file_list is None:
            self.ids = [file.stem for file in self.images_dir.iterdir() if not file.name.startswith('.')]
            self.ids = sorted([id for id in self.ids if self._get_mask_file(id).exists()])
        else:
            self.ids = sorted([Path(file).stem for file in file_list])

    def _get_mask_file(self, img_id: str) -> Path:
        return self.masks_dir / f'{img_id}_mask.jpg'
        #return self.masks_dir / f'{img_id}.jpg'
        
    def __len__(self):
        return len(self.ids)

    def get_raw_image(self, idx):
        name = self.ids[idx]
        dir = self.images_dir[idx] if isinstance(self.images_dir, list) else self.images_dir
        img_file = list(dir.glob(name + '.*'))[0]
        
        img = cv2.imread(str(img_file))
        if img is None:
            raise RuntimeError(f'Failed to load image: {img_file}')
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img

    def preprocess(self, image):
        # Convert to numpy array and normalize
        img_ndarray = np.asarray(image)
        if img_ndarray.ndim == 2:
            img_ndarray = img_ndarray[np.newaxis, ...]
        else:
            img_ndarray = img_ndarray.transpose((2, 0, 1))
        
        if img_ndarray.max() > 1:
            img_ndarray = img_ndarray / 255.0
            
        return img_ndarray

    def __getitem__(self, idx):
        name = self.ids[idx]
        
        # Load image
        img_file = self.images_dir / f'{name}.jpg'
        image = Image.open(img_file)
        
        # Load mask
        mask_file = self._get_mask_file(name)
        mask = Image.open(mask_file)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=np.array(image), mask=np.array(mask))
            image = transformed['image']
            mask = transformed['mask']
        else:
            image = np.array(image)
            mask = np.array(mask)
            
        # Preprocess
        image = self.preprocess(image)
        mask = self.preprocess(mask)
            
        return {
            'image': image.astype(np.float32),
            'mask': mask.astype(np.float32)
        }