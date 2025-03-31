import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import cv2

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str = None, mask_suffix: str = '', transform=None):
        self.images_dir = images_dir #list or single directory
        self.masks_dir = masks_dir
        self.mask_suffix = mask_suffix
        self.transform = transform

        # files from one directory only (id=filename stem)
        if isinstance(self.images_dir, list): # list of files
            self.ids = [Path(file).stem for file in self.images_dir]
            self.images_dir = [Path(file).parent for file in self.images_dir]
        elif self.images_dir.is_dir(): # directory (pick all files)
            self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        else: # single file
            self.ids = [self.images_dir.stem]
            self.images_dir = self.images_dir.parent

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, is_mask):
        img_ndarray = (pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray
    
    def get_raw_image(self, idx):
        name = self.ids[idx]
        dir = self.images_dir[idx] if isinstance(self.images_dir, list) else self.images_dir
        img_file = list(dir.glob(name + '.*'))[0]
        
        img = cv2.imread(str(img_file))
        if img is None:
            raise RuntimeError(f'Failed to load image: {img_file}')
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):      
        name = self.ids[idx]
        dir = self.images_dir[idx] if isinstance(self.images_dir, list) else self.images_dir
        img_file = list(dir.glob(name + '.*'))
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file} in {dir}'
        img = self.load(img_file[0])

        mask = None
        if self.masks_dir is not None:
            mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
            assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
            mask = self.load(mask_file[0])

            assert img.size == mask.size, \
                f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = np.asarray(img)
        if mask is not None: 
            mask = np.asarray(mask)
        
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        img = self.preprocess(img, is_mask=False)
        
        sample = {
                'image': torch.as_tensor(img.copy()).float().contiguous(),
                'name': img_file[0].name
            }

        if mask is not None: 
            mask = self.preprocess(mask, is_mask=False) #True)
            sample['mask'] = torch.as_tensor(mask.copy()).float().contiguous()

        return sample
                
