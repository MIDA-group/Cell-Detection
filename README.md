Code for:
# A Comparison of Deep Learning Methods for Cell Detection in Digital Cytology
[ArXiv pre-print] [SCIA2025] 

### Dataset
The [CNSeg Dataset](https://www.sciencedirect.com/science/article/pii/S016926072300398X) is made publcly available [on Kaggle](https://www.kaggle.com/datasets/zhaojing0522/cervical-nucleus-segmentation).

### Requisites
> pip install requisites.txt


### Segmentation-based Methods

* StarDist - Cellpose
The code to evaluate this models is available in their relative folders. Note that you need to provide the path to the data (images and labels).

* SAM2
  Similarly to the other segmentation models, SAM2 evaluation code can be found in its folder. Pre-trained SAM2 models can be found [here](https://github.com/facebookresearch/sam2).

### Centroid-based Methods
* FCRN - IFCRN
  In the respective folders, you can find the code to:
  + Build each model architecture,
  + Train-from-scratch both FRCN/IFCRN,
  + Evalaute the models.

Note that path to data, labels, and binary masks (for training) have to be passed to the parser. 
