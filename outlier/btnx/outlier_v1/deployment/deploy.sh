#!/usr/bin/env bash

# (1) Archive the model into a .mar
torch-model-archiver --model-name btnx_outlier \
                     --version 1.0 \
                     --model-file outlier_model.py \
                     --serialized-file btnx_outlier.pt \
                     --handler outlier_handler.py \
                     --extra-files kit_data.json

# (2) Make new directory for model storing and move .mar there
mkdir model_store
mv btnx_outlier.mar model_store

# (3) Start torchserve for local deployment
torchserve --start \
           --model-store model_store \
           --models btnx_outlier=btnx_outlier.mar