#!/usr/bin/env bash

# (1) Send prediction and get request
curl -X POST http://127.0.0.1:8080/predictions/maskrcnn -T oraquick_10.jpg 
