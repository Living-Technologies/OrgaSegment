#!/bin/bash

#Predicts the organoids found in the folder data using
#the provided model and saves the masks, previews, results.csv
#tracked.csv in the 'oseg500' folder.

python -m orgasegment2 -pt -m models/cellpose-oseg-500-epoch.model -d data -o oseg500

