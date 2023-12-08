# OrgaSegment
[![tests](https://github.com/Living-Technologies/OrgaSegment/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/Living-Technologies/OrgaSegment/actions)

Organoid Segmentation based on Matterport MASK-RCNN developed to segment patient derived intestinal organoids using brightfield microscopy.
Scientific manuscript submitted.  

---
## Requirements

* Linux or Windows installation
* Conda installation with [mamba installed](https://mamba.readthedocs.io/en/latest/installation.html) in base environment
* For inference a GPU is prefered but not required
* For training we recommend the use of one or multiple GPUs with >15GB RAM

This code was developed and tested on Ubuntu 20.04 and Windows

The model was trained on a HPC cluster with SLURM batch manager using a single node with 320GB memory and 4x RTX6000 GPUs

---
## Installation

Clone repository
```sh
$ git clone https://github.com/Living-Technologies/OrgaSegment.git
```

We recommend to install using mamba:
```sh
$ cd OrgaSegment
$ mamba env create -f conf/environment.yml
```

---
## Download model

* Download latest model from
* Model used for publication: [OrganoidBasic20211215.h5](https://github.com/Living-Technologies/OrgaSegment/raw/5bd6a5c45c02830f908a8bc187e3631a304d1b6c/models/OrganoidBasic20211215.h5)
* Save in /models

---
## Inference / predictions using OrgaSegment app

Start OrgaSegment app using command line
```sh
$ cd OrgaSegment
$ conda activate OrgaSegment
$ streamlit run app.py
```
Or use provided scripts:
* For linux: Run startOrgaSegmentAppLin.sh
* For windows Run startOrgaSegmentAppWin.bat

<u>Note: when running streamlit for the first time you are asked to provide contact details. You can just leave this empty using the return key (twice).</u>

### **App usage:**
* Select correct configuration (app configuration is managed in ./conf/app.conf)
* Click Inference for organoid prediction and/or Track for organoid segmentation tracking over time
* Select folder with brightfield microscopy images (currently only JPEG / JPG supported)
* Set tracking settings (if applicable)
* Run

### **Tracking settings:**
For correct organoid tracking every image should be assigned to a WELL (location / condition) and T (time). This information should be in the name of the image so it can be extracted using a REGEX.

*Example: DATETODAY_PLATE-01_WELL-A1_t0*

The correct information can be extracted using the following regular expression
```REGEX
.*(?P<WELL>[A-Z]{1}[0-9]{1,2}).*t|T(?P<T>[0-9]{1,2}).*
```
<u>The regular expression assignment for well / location should always be named WELL and the assignment for time should always be named T. Otherwise the app will throw an error.
</u>

Please, check a validate you regex using an online regular expression test tool such as [Pythex](https://pythex.org/)

In addition adjust the following settings:
* **Tracking search range in pixels:** the maximum distance in pixels organoids can move between frames
* **Memory:** the maximum number of frames during which an organoid can vanisch, then reappear within the search range, and be considered the same organoid

For more info see trackpy.link [documentation](http://soft-matter.github.io/trackpy/v0.6.1/generated/trackpy.link.html) 

---
## Inference / predictions using SLURM batch manager
Instead of running the interactive app you can also run inference on a compute cluster using SLURM batch manager.

If needed adjust the SLURM cluster setting in ./predict.sh and the configuration (including tracking settings) correct model configuration file in ./conf/

<u>Make sure you have installed the OrgaSegment conda environment on your SLURM cluster.</u>

To run inference, execute the following on the correct node of a SLURM cluster:
```sh
cd OrgaSegment
sbatch predict.sh -p -t -c conf/OrganoidBasicConfig20211215.py -f /data/folder/images/
```
predict.sh options:
* -p: inference on data
* -t: tracking on segmented organoids
* -c [config file]: location of configuration file to use with inference
* -f [data folder]: location of folder with imaging data to run inference on 

<u>Predict.sh can run only inference, only tracking or combined. A different configuration file can be selected if needed. 
</u>

## Train on HPC using SLURM batch manager
This repository always for training on your own dataset. We advice to use a High Performance Compute environment with sufficient compute power for training.

**Config**
Training requires a custom confirguration. You can use ./conf/OrganoidBasicConfig20211215.py as a basis.
A specific model (such as OrganoidBasic20211215.h5) can be used for transfered learning by setting PRETRAINED_WEIGHTS. 
<u>Make sure to review all settings.</u>

**Data**
Make sure to organize your dataset as follows:
```bash
└── data
    └── datsetName
       ├── train
       │   ├── 001_img.jpg
       │   ├── 001_masks_classA.png
       │   ├── 001_masks_classB.png
       │   ├── 002_img.jpg
       │   ├── 002_masks_classA.png
       │   ├── 002_masks_classB.png
       │   ├── 003_img.jpg
       │   ├── 003_masks_classA.png
       │   ├── 003_masks_classB.png
       |   └── etc...
       ├── val
       │   ├── 101_img.jpg
       │   ├── 101_masks_classA.png
       │   ├── 101_masks_classB.png
       │   ├── 102_img.jpg
       │   ├── 102_masks_classA.png
       │   ├── 102_masks_classB.png
       |   └── etc...
       └── eval
           ├── 201_img.jpg
           ├── 201_masks_classA.png
           ├── 201_masks_classB.png
           └── etc...
```
To take under consideration:
* The image name should contain _img otherwise change config
* The mask should contain \_masks\_ otherwise change config
* Every class (in a multiclass clasification) should contain its own mask file
* The class names in the mask filename should correspond to the class names in the config file
* A mask file is an array where 0 is background and each (pixel) value >0 is a unique mask. So a mask array with 2 masks contains multiple values of only 0, 1 and 2.
* A typical data set contains unique images and masks for training, validation and evalutation/testing.

**Run**
If needed adjust the SLURM cluster setting in ./train.sh

<u>Make sure you have installed the OrgaSegment conda environment on your SLURM cluster.</u>

To run inference, execute the following on the correct node of a SLURM cluster:
```sh
cd OrgaSegment
sbatch train.sh -t -e -c conf/OrganoidBasicConfig20211215.py -m /models/ABC.h5
```
train.sh options:
* -p: train on data
* -e: evaluate/test trained model 
* -c [config file]: location of configuration file to use with itraining
* -m [model file]: location of model to evaluate/test. If -m is not used but -e is active, the latest trained model will be picked to evaluate (typical use of training and evaluation is done in one run).

<u>Train.sh can run only tracking, only evaluation or combined. A different configuration file can be selected if needed. 
</u>

## ToDo
* Support Tensorflow 2
* Support Python >3.6
* Support for Neptune model monitoring

## Credits
* [Labelbox](https://labelbox.com/) academic license use
* [Matterport MASK-RCNN](https://github.com/matterport/Mask_RCNN) repository
* [Cellpose](https://github.com/MouseLand/cellpose) Average precision fucntion 