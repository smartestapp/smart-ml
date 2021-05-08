#!/usr/bin/env bash

# (1) Archive the model into a .mar
torch-model-archiver --model-name maskrcnn \
                     --version 1.0 \
                     --model-file maskrcnn_model.py \
                     --serialized-file ../saved_models/maskrcnn_weights.pth \
                     --handler maskrcnn_handler.py \
                     --extra-files index_to_name.json

# (2) Make new directory for model storing and move .mar there
mkdir model_store
mv maskrcnn.mar model_store

# (3) Start torchserve for local deployment
torchserve --start \
           --model-store model_store \
           --models maskrcnn=maskrcnn.mar
