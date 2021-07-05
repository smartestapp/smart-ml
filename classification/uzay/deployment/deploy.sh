#!/usr/bin/env bash

# (1) Archive the model into a .mar
torch-model-archiver --model-name oraquick_classifier \
                     --version 1.0 \
                     --model-file classifier_model.py \
                     --serialized-file oraquick_classifier.pt \
                     --handler classifier_handler.py \
                     --extra-files kit_data.json

# (2) Make new directory for model storing and move .mar there
mkdir model_store
mv oraquick_classifier.mar model_store

# (3) Start torchserve for local deployment
torchserve --start \
           --model-store model_store \
           --models oraquick_classifier.mar