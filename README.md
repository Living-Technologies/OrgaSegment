# OrgaSegment
[![tests](https://github.com/Living-Technologies/OrgaSegment/workflows/tests/badge.svg?branch=master)](https://github.com/Living-Technologies/OrgaSegment/actions)

Organoid Segmentation based on Matterport MASK-RCNN developed to segment patient derived intestinal organoids using brightfield microscopy.
Scientific manuscript submitted.  

## Requirements

* Linux or Windows installation
* Conda installation with [mamba installed](https://mamba.readthedocs.io/en/latest/installation.html) in base environment
* For interference a GPU is prefered but not required
* For training we recommend the use of one or multiple GPUs with >15GB RAM

This code was developed and tested on Ubuntu 20.04 and Windows

The model was trained on a HPC cluster with SLURM batch manager using a single node with 320GB memory and 4x RTX6000 GPUs

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

## Download model

* Download latest model from
* Model used for publication: OrganoidBasic20211215.h5
* Save in /models

## Interference / predictions using OrgaSegment app

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

App usage:
* Select correct configuration (app configuration is managed in ./conf/app.conf)
* Click Interference for organoid prediction and/or Track for organoid segmentation tracking over time
* Select folder with brightfield microscopy images (currently only JPEG / JPG supported)


## Interference / predictions using SLURM batch manager



## Train on HPC using SLURM batch manager


## ToDo
* Support Tensorflow 2
* Support Python >3.6
* Support for Neptune model monitoring

## Credits
