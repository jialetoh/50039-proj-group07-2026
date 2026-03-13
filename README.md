# Industrial Cable Defect Detection using Autoencoders

SUTD Project for 50.039 Theory and Practice of Deep Learning (2026)

#### Group 7
- Ang Li En Eldrick (1006908)
- Malvin Ken Sudirgo (1007164)
- Toh Jia Le (1007004)


## Overview

This project does pixel-level anomaly detection and binary segmentation of cable defects, for industrial quality control and automated inspection.


## Dataset

This project uses the cable category from the MVTec Anomaly Detection (MVTec AD) dataset, downloadable from Kaggle [here]](https://www.kaggle.com/datasets/ipythonx/mvtec-ad?select=cable). More information available at the [official MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad).

Download the dataset and place the `cable/` folder inside the `data/` folder (see Project Directory below).


## Project Directory

```text
├── README.md
├── requirements.txt
├── .gitignore
├── data/ 						# put dataset here
│   └── cable/
│       ├── train/
│       ├── test/
│       └── ground_truth/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_autoencoder.ipynb
│   ├── 03_pretrained_encoder.ipynb
│   ├── 04_augmentation_and_tuning.ipynb
│   └── 05_final_report.ipynb
├── src/
│   ├── dataset.py 			# dataset loading & preprocessing
│   ├── models.py 			# model definitions
│   ├── train.py  			    # model training
│   ├── eval.py  			    # model evaluation
│   ├── metrics.py 			# evaluation metrics
│   └── utils.py 					# helper functions      
├── checkpoints/ 				# saved weights 
├── outputs/						# generated outputs
│   ├── figures/
│   └── sample_outputs/
└── pdf/                            # pdf files for submission
```

## Project Setup

1. Install required dependencies `pip install -r requirements.txt`
2. Download the dataset and place it in the correct folder (see Dataset section)
3. Run the notebooks in sequence

### Requirements
- Python 3 (tested on v3.13.4)
- OpenCV (tested on 4.13.0)
- PyTorch (tested on 2.10.0)
- Matplotlib (tested on v3.10.8)
- Numpy (tested on v2.4.1)


## References

- Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger, "A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", IEEE Conference on Computer Vision and Pattern Recognition, 2019