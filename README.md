Code for:
# A Comparison of Deep Learning Methods for Cell Detection in Digital Cytology
## M. Acerbis, N. Sladoje, and J. Lindblad
(https://arxiv.org/abs/2504.06957)[ArXiv pre-print] [SCIA2025] 

### Dataset
The [CNSeg Dataset](https://www.sciencedirect.com/science/article/pii/S016926072300398X) is made publicly available [on Kaggle](https://www.kaggle.com/datasets/zhaojing0522/cervical-nucleus-segmentation).

### Requisites
```
pip install requisites.txt
```

### Segmentation-based Methods

* StarDist: modify the requested paths to data and labels. Then run:
 ```
python stardist_test.py
```
* Cellpose: modify the requested paths to data and labels. Then run:
 ```
python cellpose_test.py
```
* SAM2: pre-trained SAM2 models can be found [here](https://github.com/facebookresearch/sam2). Modify the requested paths to data and labels. Then run:
```
python SAM2_test.py
```

### Centroid-based Methods
* FCRN - IFCRN
  In the respective folders, you can find the code to:
  + Build each model architecture,
  + Train-from-scratch\
    FRCN: modify the requested paths and then run
    ```
    python fcrn_train.py
    ```
    and IFCRN: run the following
    ```
    python train_cnseg.py --train-img-dir --train-mask-dir --val-img-dir --val-mask-dir -e -b 
    ```
  + Evalaute the models\
    FRCN: modify the requested paths and then run
    ```
    python fcrn_test.py
    ```
    and IFCRN: run the following
    ```
    python predict.py -i -o -m -t --labels-dir
    ```

Note that path to data, labels, and binary masks (for training) have to be passed to the parser. 
