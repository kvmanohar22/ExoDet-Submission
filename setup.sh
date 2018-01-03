#!/bin/bash

echo "Installing dependencies..."
sudo apt-get -qq --yes update
sudo apt-get --yes install python-pip
sudo pip install -r requirements.txt

# TODO: Download the dataset from google drive and place it in `data` directory
mkdir -p results probs model_params
