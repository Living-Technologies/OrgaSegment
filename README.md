# OrgaSegment
[![tests](https://github.com/Living-Technologies/OrgaSegment/workflows/tests/badge.svg?branch=master)](https://github.com/Living-Technologies/OrgaSegment/actions)

Organoid Segmentation based on Matterport MASK-RCNN developed to segment patient derived intestinal organoids using brightfield microscopy.
Scientific manuscript submitted.  


>>> Image ![Overview](images/plot.png)
## Requirements

* Linux or Windows installation
* Conda installation with [mamba installed](https://mamba.readthedocs.io/en/latest/installation.html) in base environment
* For interference a GPU is prefered but not required
* For training we recommend the use og one or multiple GPUs with >15GB RAM

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

Save in /models

## Run OrgaSegment app


## Train


##ToDo
* Support Tensorflow 2
* Support Python >3.6
* Support for Neptune model monitoring

## Credits
