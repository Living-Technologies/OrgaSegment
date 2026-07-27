# OrgaSegment
[![tests](https://github.com/Living-Technologies/OrgaSegment/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/Living-Technologies/OrgaSegment/actions)

Organoid Segmentation based on [cellpose](https://www.cellpose.org/) developed to segment patient derived intestinal organoids using brightfield microscopy published on [nature.com](https://www.nature.com/articles/s42003-024-05966-4), March 2024.

---
## Requirements

* Linux or Windows installation
* Conda installation with [mamba installed](https://mamba.readthedocs.io/en/latest/installation.html) in base environment
* For inference a GPU is prefered but not required
* For training we recommend the use of one or multiple GPUs with >8GB RAM

This code was developed and tested on Ubuntu 20.04 and Windows 10, 11 with wsl.

---
## Installation

Clone this repository with git or download it. 

We recommend to install using mamba:

```sh
$ cd OrgaSegment2	
$ mamba env create -f conf/environment.yml
```

### Download the latest model

The most recent cellpose model is available upon request. If none is provided a default cellpose model will be used.

### Alternative Installation.

Any python virtual environment tool should work, install the dependencies found in environment.yml. As long as the environment is
activated, you can run orgasegment2. This has the advantage that any python management tool can be used. Orgasegment2
can also be installed into the environment, then it can be run from anywhere. Also different versions of cellpose can
be used.

orgasegment2 can also be installed in the virtual environment. Then `streamlit run /path/to/app.py` can be run from anywhere
with the conf folder.

## **App usage:**
* Select correct configuration (app configuration is managed in ./conf/appCellPose.conf)
* Click Inference for organoid prediction and/or Track for organoid segmentation tracking over time
* Enter the path to the folder with images to be processed. This can be an absolute path, or a path relative the app.py
* Set tracking settings (if applicable)
* Run

### Inference / predictions using OrgaSegment app

Start OrgaSegment app using command line
```sh
$ cd OrgaSegment
$ conda activate OrgaSegment
$ streamlit run app.py
```

**To use the GPU you must set an environment variable before starting.**
```sh
$ cd OrgaSegment
$ conda activate OrgaSegment
$ export USE_GPU=True
$ streamlit run app.py
```

Or use provided scripts:
* For linux: Run startOrgaSegmentAppLin.sh
* For windows Run startOrgaSegmentAppWin.bat

<u>Note: when running streamlit for the first time you are asked to provide contact details. You can just leave this empty using the return key (twice).</u>




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


## Train

We train cellpose with the following command:

    python -m cellpose --train \
      --dir "$(realpath $train_dir)" --use_gpu --test_dir "$(realpath $val_dir)" \
      --img_filter $img_filter --mask_filter $mask_filter \
      --pretrained_model $pm --verbose

The parameters we pass to cellpose:

    $train_dir   # location of the training data.
    $val_dir     # location of the validation data.
	$img_filter  # is "\_img"  
    $mask_filter # is "\_masks\_organoid" 
    $pm          # pretrained model. 	

We've included a script "train_cellpose.sh" where you can train cellpose as follows.

        ./train_cellpose.sh pretrained_model data_folder
    
*That assumes that the orgasment2 conda environment is activated.*

Another example script [train_original.sh](scripts/train_original.sh) will train an new cellpose model on the [published segmentation data](https://www.nature.com/articles/s42003-024-05966-4). That can be useful to debug before trying
with new data.

**Data**
The dataset is organized as follows:
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
* A mask file is an array where 0 is background and each (pixel) value > 0 is a unique mask. So a mask array with 2 masks contains multiple values of only 0, 1 and 2. *


## Organoids Mask Segmentation metrics

This is a small is a set of results based on the original orgasegment dataset, [Organoids Basic](https://zenodo.org/records/10278229). 

### Leader board

| technique | masks | average mask JI | background JI | TP | FN | FP |
|:----------|:-----:|:---------------:|:-------------:|---:|---:|---:|
|Orgasegment|973    | 0.766           | 0.985         | 755| 182| 46 |
|DT - 2D    |973    | 0.424           | 0.956         | 631| 333| 17 |
|Cellpose   |973    | 0.787           | 0.982         | 843| 103| 32 |
|Sam - NT   |973    | 0.670           | 0.967         | 486| 382| 117|
|CP SAM     |973    | 0.800           | 0.987         | 820| 137| 17 |

The evaluations were performed using code found [OsegKaggle](https://github.com/Living-Technologies/OsegKaggle).



## Credits
* [Labelbox](https://labelbox.com/) academic license use
* [Cellpose](https://github.com/MouseLand/cellpose) Average precision fucntion 
